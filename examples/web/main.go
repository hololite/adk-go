// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"log"
	"os"

	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/examples/lib/geminihelper"
	"google.golang.org/adk/examples/lib/ollama"
	"google.golang.org/adk/examples/web/agents"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

func saveReportfunc(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
	if llmResponse == nil || llmResponse.Content == nil || llmResponseError != nil {
		return llmResponse, llmResponseError
	}
	for _, part := range llmResponse.Content.Parts {
		if part.Text == "" && part.InlineData == nil {
			continue
		}
		_, err := ctx.Artifacts().Save(ctx, uuid.NewString(), part)
		if err != nil {
			return nil, err
		}
	}
	return llmResponse, llmResponseError
}

// AuthInterceptor sets 'user' name needed for both a2a and webui launchers which sharing the same sessions service.
type AuthInterceptor struct {
	a2asrv.PassthroughCallInterceptor
}

// Before implements a before request callback.
func (a *AuthInterceptor) Before(ctx context.Context, callCtx *a2asrv.CallContext, req *a2asrv.Request) (context.Context, error) {
	callCtx.User = &a2asrv.AuthenticatedUser{
		UserName: "user",
	}
	return ctx, nil
}

func main() {
	ctx := context.Background()
	apiKey := os.Getenv("GOOGLE_API_KEY")

	var llm model.LLM
	var err error
	if ollama.IsEnabled() {
		llm, err = ollama.NewModel(ctx, ollama.ModelName())
	} else {
		llm, err = gemini.NewModel(ctx, geminihelper.ModelName(), &genai.ClientConfig{
			APIKey: apiKey,
		})
	}
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	if !ollama.IsEnabled() {
		log.Printf("gemini: using model %q", geminihelper.ModelName())
	}

	var weatherTools []tool.Tool
	if ollama.IsEnabled() {
		searchTool, err := ollama.NewSearchTool(ctx)
		if err != nil {
			log.Fatalf("Failed to create search tool: %v", err)
		}
		weatherTools = []tool.Tool{searchTool}
	} else {
		weatherTools = []tool.Tool{geminitool.GoogleSearch{}}
	}

	sessionService := session.InMemoryService()
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       llm,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Tools:       weatherTools,
		AfterModelCallbacks: []llmagent.AfterModelCallback{saveReportfunc},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	llmAuditor := agents.GetLLMAuditorAgent(ctx, llm)
	imageGeneratorAgent := agents.GetImageGeneratorAgent(ctx, llm)

	agentLoader, err := agent.NewMultiLoader(
		rootAgent,
		llmAuditor,
		imageGeneratorAgent,
	)
	if err != nil {
		log.Fatalf("Failed to create agent loader: %v", err)
	}

	artifactservice := artifact.InMemoryService()

	config := &launcher.Config{
		ArtifactService: artifactservice,
		SessionService:  sessionService,
		AgentLoader:     agentLoader,
		A2AOptions: []a2asrv.RequestHandlerOption{
			a2asrv.WithCallInterceptor(&AuthInterceptor{}),
		},
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
