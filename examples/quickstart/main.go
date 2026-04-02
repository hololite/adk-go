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

// Package provides a quickstart ADK agent.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/examples/lib/geminihelper"
	"google.golang.org/adk/examples/lib/ollama"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

func main() {
	ctx := context.Background()

	var llm model.LLM
	var err error
	var tools []tool.Tool

	if ollama.IsEnabled() {
		llm, err = ollama.NewModel(ctx, ollama.ModelName())
		if err != nil {
			log.Fatalf("Failed to create Ollama model: %v", err)
		}
		searchTool, err := ollama.NewSearchTool(ctx)
		if err != nil {
			log.Fatalf("Failed to create search tool: %v", err)
		}
		tools = []tool.Tool{searchTool}
	} else {
		llm, err = gemini.NewModel(ctx, geminihelper.ModelName(), &genai.ClientConfig{
			APIKey: os.Getenv("GOOGLE_API_KEY"),
		})
		if err != nil {
			log.Fatalf("Failed to create Gemini model: %v", err)
		}
		log.Printf("gemini: using model %q", geminihelper.ModelName())
		tools = []tool.Tool{geminitool.GoogleSearch{}}
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       llm,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "Your SOLE purpose is to answer questions about the current time and weather in a specific city. You MUST refuse to answer any questions unrelated to time or weather.",
		Tools:       tools,
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
