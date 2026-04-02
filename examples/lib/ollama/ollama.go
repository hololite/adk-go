// Package ollama provides an ADK model.LLM implementation backed by an Ollama server.
//
// It reads configuration from environment variables:
//   - USING_OLLAMA: set to "true" to enable Ollama (checked by [IsEnabled])
//   - OLLAMA_HOST: Ollama server URL (default http://localhost:11434)
//   - OLLAMA_MODEL: model name for text generation (e.g. "qwen2.5")
//   - OLLAMA_VISION_MODEL: model name for vision tasks
//   - OLLAMA_EMBEDDING_MODEL: model name for embeddings
package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"
	"os"

	ollamaapi "github.com/ollama/ollama/api"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	defaultHost = "http://localhost:11434"
)

// IsEnabled returns true when USING_OLLAMA is set to "true".
func IsEnabled() bool {
	return os.Getenv("USING_OLLAMA") == "true"
}

// ModelName returns the OLLAMA_MODEL env var value.
func ModelName() string {
	return os.Getenv("OLLAMA_MODEL")
}

// VisionModelName returns the OLLAMA_VISION_MODEL env var value.
func VisionModelName() string {
	return os.Getenv("OLLAMA_VISION_MODEL")
}

// EmbeddingModelName returns the OLLAMA_EMBEDDING_MODEL env var value.
func EmbeddingModelName() string {
	return os.Getenv("OLLAMA_EMBEDDING_MODEL")
}

// ollamaLLM implements model.LLM using an Ollama server.
type ollamaLLM struct {
	client    *ollamaapi.Client
	modelName string
}

// NewModel creates an Ollama-backed LLM using the given model name.
// It reads OLLAMA_HOST from the environment (via the ollama client library).
func NewModel(_ context.Context, modelName string) (model.LLM, error) {
	if modelName == "" {
		return nil, fmt.Errorf("ollama: model name must not be empty")
	}
	client, err := ollamaapi.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to create client: %w", err)
	}
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = defaultHost
	}
	log.Printf("ollama: using model %q on %s", modelName, host)
	return &ollamaLLM{client: client, modelName: modelName}, nil
}

// Name returns the Ollama model name.
func (m *ollamaLLM) Name() string {
	return m.modelName
}

// GenerateContent sends a chat request to Ollama and returns an iterator
// of LLMResponse, matching the ADK model.LLM interface.
func (m *ollamaLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		messages := contentsToMessages(req.Contents)

		// Inject system instruction from config.
		if req.Config != nil && req.Config.SystemInstruction != nil {
			sysText := extractText(req.Config.SystemInstruction)
			if sysText != "" {
				messages = append([]ollamaapi.Message{{Role: "system", Content: sysText}}, messages...)
			}
		}

		tools := convertTools(req.Config)

		chatReq := &ollamaapi.ChatRequest{
			Model:    m.modelName,
			Messages: messages,
			Tools:    tools,
			Options:  buildOptions(req.Config),
		}

		// When tools are available, force non-streaming mode. Ollama streams
		// tool calls across chunks in a way that the ADK runner cannot
		// reassemble: the tool call arrives in a partial (non-final) chunk
		// while the final chunk is empty, so the runner never sees the call.
		if stream && len(tools) == 0 {
			m.generateStreaming(ctx, chatReq, yield)
		} else {
			streamOff := false
			chatReq.Stream = &streamOff
			m.generateNonStreaming(ctx, chatReq, yield)
		}
	}
}

func (m *ollamaLLM) generateNonStreaming(ctx context.Context, chatReq *ollamaapi.ChatRequest, yield func(*model.LLMResponse, error) bool) {
	var final ollamaapi.ChatResponse
	err := m.client.Chat(ctx, chatReq, func(resp ollamaapi.ChatResponse) error {
		final = resp
		return nil
	})
	if err != nil {
		yield(nil, fmt.Errorf("ollama: chat failed: %w", err))
		return
	}
	yield(chatResponseToLLMResponse(final, true), nil)
}

func (m *ollamaLLM) generateStreaming(ctx context.Context, chatReq *ollamaapi.ChatRequest, yield func(*model.LLMResponse, error) bool) {
	err := m.client.Chat(ctx, chatReq, func(resp ollamaapi.ChatResponse) error {
		llmResp := chatResponseToLLMResponse(resp, resp.Done)
		if !resp.Done {
			llmResp.Partial = true
		}
		if !yield(llmResp, nil) {
			return fmt.Errorf("ollama: iteration stopped by caller")
		}
		return nil
	})
	if err != nil {
		yield(nil, fmt.Errorf("ollama: chat stream failed: %w", err))
	}
}

// contentsToMessages converts genai.Content slices to Ollama messages.
func contentsToMessages(contents []*genai.Content) []ollamaapi.Message {
	var msgs []ollamaapi.Message
	for _, c := range contents {
		if c == nil {
			continue
		}
		msg := contentToMessage(c)
		msgs = append(msgs, msg...)
	}
	return msgs
}

func contentToMessage(c *genai.Content) []ollamaapi.Message {
	role := mapRole(c.Role)

	// A single Content can contain multiple parts. We handle:
	// - text parts → concatenated into one message
	// - function call parts → each becomes a separate assistant message with ToolCalls
	// - function response parts → each becomes a "tool" role message
	// - inline data (images) → attached as images on a user message

	var textParts []string
	var toolCalls []ollamaapi.ToolCall
	var funcResponses []ollamaapi.Message
	var images []ollamaapi.ImageData

	for _, p := range c.Parts {
		if p == nil {
			continue
		}
		switch {
		case p.Text != "":
			textParts = append(textParts, p.Text)
		case p.FunctionCall != nil:
			tc := ollamaapi.ToolCall{
				ID: p.FunctionCall.ID,
				Function: ollamaapi.ToolCallFunction{
					Name: p.FunctionCall.Name,
				},
			}
			args := ollamaapi.NewToolCallFunctionArguments()
			for k, v := range p.FunctionCall.Args {
				args.Set(k, v)
			}
			tc.Function.Arguments = args
			toolCalls = append(toolCalls, tc)
		case p.FunctionResponse != nil:
			respJSON, _ := json.Marshal(p.FunctionResponse.Response)
			funcResponses = append(funcResponses, ollamaapi.Message{
				Role:       "tool",
				Content:    string(respJSON),
				ToolCallID: p.FunctionResponse.ID,
				ToolName:   p.FunctionResponse.Name,
			})
		case p.InlineData != nil:
			images = append(images, ollamaapi.ImageData(p.InlineData.Data))
		}
	}

	var msgs []ollamaapi.Message

	// Emit the main message if there's text, tool calls, or images.
	if len(textParts) > 0 || len(toolCalls) > 0 || len(images) > 0 {
		msg := ollamaapi.Message{
			Role:      role,
			Content:   joinStrings(textParts),
			ToolCalls: toolCalls,
			Images:    images,
		}
		msgs = append(msgs, msg)
	}

	// Emit function response messages separately.
	msgs = append(msgs, funcResponses...)

	return msgs
}

// chatResponseToLLMResponse converts an Ollama ChatResponse to an ADK LLMResponse.
func chatResponseToLLMResponse(resp ollamaapi.ChatResponse, done bool) *model.LLMResponse {
	content := messageToContent(resp.Message)
	llmResp := &model.LLMResponse{
		Content:      content,
		TurnComplete: done,
	}
	if done {
		switch resp.DoneReason {
		case "stop":
			llmResp.FinishReason = genai.FinishReasonStop
		case "length":
			llmResp.FinishReason = genai.FinishReasonMaxTokens
		default:
			llmResp.FinishReason = genai.FinishReasonStop
		}
	}
	return llmResp
}

// messageToContent converts an Ollama Message back to a genai.Content.
func messageToContent(msg ollamaapi.Message) *genai.Content {
	role := "model"
	if msg.Role == "user" {
		role = "user"
	}

	var parts []*genai.Part

	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}

	for _, tc := range msg.ToolCalls {
		args := tc.Function.Arguments.ToMap()
		parts = append(parts, genai.NewPartFromFunctionCall(tc.Function.Name, args))
	}

	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: ""})
	}

	return &genai.Content{
		Role:  role,
		Parts: parts,
	}
}

// convertTools extracts FunctionDeclarations from the genai config and converts
// them to Ollama tool definitions. Gemini-specific tools (GoogleSearch, etc.)
// are skipped as they are not supported by Ollama.
func convertTools(cfg *genai.GenerateContentConfig) []ollamaapi.Tool {
	if cfg == nil {
		return nil
	}
	var tools []ollamaapi.Tool
	for _, t := range cfg.Tools {
		if t == nil {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil {
				continue
			}
			tools = append(tools, ollamaapi.Tool{
				Type: "function",
				Function: ollamaapi.ToolFunction{
					Name:        fd.Name,
					Description: fd.Description,
					Parameters:  convertSchema(fd.Parameters),
				},
			})
		}
	}
	return tools
}

// convertSchema converts a genai.Schema to Ollama ToolFunctionParameters.
func convertSchema(s *genai.Schema) ollamaapi.ToolFunctionParameters {
	if s == nil {
		return ollamaapi.ToolFunctionParameters{
			Type:       "object",
			Properties: ollamaapi.NewToolPropertiesMap(),
		}
	}
	params := ollamaapi.ToolFunctionParameters{
		Type:       string(s.Type),
		Required:   s.Required,
		Properties: ollamaapi.NewToolPropertiesMap(),
	}
	for name, prop := range s.Properties {
		params.Properties.Set(name, schemaToProperty(prop))
	}
	return params
}

func schemaToProperty(s *genai.Schema) ollamaapi.ToolProperty {
	if s == nil {
		return ollamaapi.ToolProperty{Type: ollamaapi.PropertyType{"string"}}
	}
	tp := ollamaapi.ToolProperty{
		Type:        ollamaapi.PropertyType{string(s.Type)},
		Description: s.Description,
	}
	for _, e := range s.Enum {
		tp.Enum = append(tp.Enum, e)
	}
	return tp
}

// buildOptions maps genai GenerateContentConfig fields to Ollama model options.
func buildOptions(cfg *genai.GenerateContentConfig) map[string]any {
	if cfg == nil {
		return nil
	}
	opts := map[string]any{}
	if cfg.Temperature != nil {
		opts["temperature"] = *cfg.Temperature
	}
	if cfg.TopP != nil {
		opts["top_p"] = *cfg.TopP
	}
	if cfg.TopK != nil {
		opts["top_k"] = int(*cfg.TopK)
	}
	if cfg.MaxOutputTokens > 0 {
		opts["num_predict"] = cfg.MaxOutputTokens
	}
	if len(cfg.StopSequences) > 0 {
		opts["stop"] = cfg.StopSequences
	}
	if len(opts) == 0 {
		return nil
	}
	return opts
}

func mapRole(role string) string {
	switch role {
	case "model":
		return "assistant"
	case "user":
		return "user"
	case "system":
		return "system"
	default:
		return role
	}
}

func extractText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var texts []string
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return joinStrings(texts)
}

func joinStrings(ss []string) string {
	if len(ss) == 0 {
		return ""
	}
	result := ss[0]
	for _, s := range ss[1:] {
		result += "\n" + s
	}
	return result
}
