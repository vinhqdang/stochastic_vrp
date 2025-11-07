{
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test:*)",
      "Read(~/.zshrc)"
    ],
    "deny": [
      "Bash(curl:*)"
    ]
  },
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "ANTHROPIC_BEDROCK_BASE_URL": "https://api.prd.tymex.cloud/aix/claude-code/1.0.0/bedrock/",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1",
    "ANTHROPIC_MODEL": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "ANTHROPIC_SMALL_FAST_MODEL": "global.anthropic.claude-sonnet-4-20250514-v1:0"
  },
  "apiKeyHelper": "${HOME}/.claude/claude_code_auth.sh",
  "includeCoAuthoredBy": false
}
