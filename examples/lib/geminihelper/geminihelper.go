// Package geminihelper provides shared helpers for configuring Gemini models
// across ADK example apps.
//
// Environment variables:
//   - GEMINI_MODEL: model name override (default "gemini-2.5-flash")
package geminihelper

import (
	"os"
)

const defaultModel = "gemini-2.5-flash"

// ModelName returns the GEMINI_MODEL env var value, defaulting to "gemini-2.5-flash".
func ModelName() string {
	if m := os.Getenv("GEMINI_MODEL"); m != "" {
		return m
	}
	return defaultModel
}
