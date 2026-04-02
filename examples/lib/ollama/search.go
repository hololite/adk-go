package ollama

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/examples/lib/geminihelper"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// SearchInput is the parameter schema for the web search tool.
type SearchInput struct {
	Query string `json:"query"`
}

// SearchOutput is the result returned by the web search tool.
type SearchOutput struct {
	Results string `json:"results"`
}

// NewSearchTool creates a function tool that performs a Google Search via the
// Gemini API with GoogleSearch grounding. This gives Ollama models access to
// real-time web results without needing a separate search API key.
//
// It requires GOOGLE_API_KEY (or GEMINI_API_KEY) to be set.
func NewSearchTool(ctx context.Context) (tool.Tool, error) {
	log.Printf("ollama/search: using Gemini model %q for search grounding", geminihelper.ModelName())
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  os.Getenv("GOOGLE_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("ollama/search: failed to create Gemini client: %w", err)
	}

	handler := func(_ tool.Context, input SearchInput) (SearchOutput, error) {
		resp, err := client.Models.GenerateContent(ctx, geminihelper.ModelName(), []*genai.Content{
			genai.NewContentFromText(input.Query, "user"),
		}, &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
		})
		if err != nil {
			return SearchOutput{}, fmt.Errorf("search failed: %w", err)
		}
		return SearchOutput{Results: formatSearchResponse(resp)}, nil
	}

	return functiontool.New(functiontool.Config{
		Name:        "google_search",
		Description: "Search the web using Google Search. Use this to find real-time information such as current weather, news, sports scores, or any up-to-date facts.",
	}, handler)
}

// formatSearchResponse extracts text and grounding references from the Gemini response.
func formatSearchResponse(resp *genai.GenerateContentResponse) string {
	if resp == nil || len(resp.Candidates) == 0 {
		return "No search results found."
	}

	var b strings.Builder

	// Extract the model's grounded answer.
	candidate := resp.Candidates[0]
	if candidate.Content != nil {
		for _, p := range candidate.Content.Parts {
			if p != nil && p.Text != "" {
				b.WriteString(p.Text)
				b.WriteByte('\n')
			}
		}
	}

	// Append source references from grounding metadata.
	if candidate.GroundingMetadata != nil && len(candidate.GroundingMetadata.GroundingChunks) > 0 {
		b.WriteString("\nSources:\n")
		for _, chunk := range candidate.GroundingMetadata.GroundingChunks {
			if chunk.Web != nil {
				b.WriteString("- ")
				if chunk.Web.Title != "" {
					b.WriteString(chunk.Web.Title)
					b.WriteString(": ")
				}
				b.WriteString(chunk.Web.URI)
				b.WriteByte('\n')
			}
		}
	}

	return b.String()
}
