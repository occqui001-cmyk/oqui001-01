// server.js - NVIDIA NIM Proxy (OpenAI-compatible)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Toggles
const SHOW_REASONING = false;        // true면 <think> 태그로 추론 과정 노출
const ENABLE_THINKING_MODE = true;  // true면 모델에 맞는 chat_template_kwargs를 자동 주입
const LOG_RESPONSES = true;          // 디버깅용. 요청/응답 둘 다 찍음

// 모델별 thinking 파라미터명이 다름
// - GLM 계열: enable_thinking (+ NVIDIA 공식 예제는 clear_thinking까지 함께 전달)
// - 그 외 (qwen, deepseek 등): thinking
function getThinkingKwargs(model = '') {
  const m = model.toLowerCase();
  if (m.includes('glm') || m.startsWith('z-ai/')) {
    return { enable_thinking: true, clear_thinking: false };
  }
  return { thinking: true };
}

// axios 에러에서 NIM이 돌려준 실제 에러 body를 뽑아내는 헬퍼
async function extractNimError(error) {
  const data = error.response?.data;
  if (!data) return null;
  if (typeof data.on === 'function') {
    try {
      const chunks = [];
      for await (const chunk of data) chunks.push(chunk);
      return JSON.parse(Buffer.concat(chunks).toString());
    } catch {
      return null;
    }
  }
  return data;
}

async function handleError(res, error, label) {
  const status = error.response?.status || 500;
  const nimError = await extractNimError(error);
  console.error(`${label}:`, status, nimError || error.message);

  if (res.headersSent) return res.end();

  if (nimError) return res.status(status).json(nimError);
  return res.status(status).json({
    error: {
      message: error.message || 'Internal server error',
      type: 'invalid_request_error',
      code: status
    }
  });
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    log_responses: LOG_RESPONSES
  });
});

// List models — NIM 쪽으로 그대로 프록시
app.get('/v1/models', async (req, res) => {
  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}` }
    });
    res.json(response.data);
  } catch (error) {
    await handleError(res, error, 'List models error');
  }
});

// Chat completions (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const nimRequest = {
      ...req.body,
      temperature: req.body.temperature ?? 0.6,
      max_tokens: req.body.max_tokens ?? 9024,
      stream: !!req.body.stream,
      ...(ENABLE_THINKING_MODE && {
        chat_template_kwargs: getThinkingKwargs(req.body.model)
      })
    };

    const stream = nimRequest.stream;

    // 🔍 NIM으로 나가는 요청 로깅 (messages는 너무 길어서 길이만 표시)
    if (LOG_RESPONSES) {
      const logPayload = {
        ...nimRequest,
        messages: `<${nimRequest.messages?.length || 0} messages>`
      };
      console.log(`\n===== [REQUEST] 프록시 → NIM =====`);
      console.log(JSON.stringify(logPayload, null, 2));
      console.log(`chat_template_kwargs 포함 여부: ${nimRequest.chat_template_kwargs ? 'YES' : 'NO'}`);
      console.log(`===== [REQUEST] 끝 =====`);
    }

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      // 🔍 디버깅용 누적 버퍼
      let accContent = '';
      let accReasoning = '';
      let chunkCount = 0;

      if (LOG_RESPONSES) {
        console.log(`\n===== [STREAM RESPONSE] model=${nimRequest.model} 시작 =====`);
      }

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const { reasoning_content: reasoning, content } = delta;

              if (LOG_RESPONSES && (reasoning || content)) {
                chunkCount++;
                if (reasoning) accReasoning += reasoning;
                if (content) accContent += content;
              }

              if (SHOW_REASONING) {
                let combined = '';
                if (reasoning && !reasoningStarted) {
                  combined = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combined = reasoning;
                }
                if (content && reasoningStarted) {
                  combined += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combined += content;
                }
                if (combined) delta.content = combined;
              }
              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => {
        if (LOG_RESPONSES) {
          console.log(`총 청크 수: ${chunkCount}`);
          console.log(`▼ content (stringify):`);
          console.log(JSON.stringify(accContent));
          console.log(`▼ reasoning_content (stringify):`);
          console.log(JSON.stringify(accReasoning));
          console.log(`▼ content (raw):`);
          console.log(accContent);
          console.log(`===== [STREAM RESPONSE] 종료 =====\n`);
        }
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      const data = response.data;

      if (LOG_RESPONSES) {
        console.log(`\n===== [NON-STREAM RESPONSE] model=${nimRequest.model} =====`);
        data.choices.forEach((choice, i) => {
          console.log(`▼ choice[${i}].message.content (stringify):`);
          console.log(JSON.stringify(choice.message?.content));
          console.log(`▼ choice[${i}].message.reasoning_content (stringify):`);
          console.log(JSON.stringify(choice.message?.reasoning_content));
          console.log(`▼ choice[${i}].message.content (raw):`);
          console.log(choice.message?.content);
        });
        console.log(`===== [NON-STREAM RESPONSE] 종료 =====\n`);
      }

      data.choices.forEach(choice => {
        if (!choice.message) return;
        if (SHOW_REASONING && choice.message.reasoning_content) {
          choice.message.content =
            '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' +
            (choice.message.content || '');
        }
        delete choice.message.reasoning_content;
      });
      res.json(data);
    }
  } catch (error) {
    await handleError(res, error, 'Proxy error');
  }
});

// 404 for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Response logging: ${LOG_RESPONSES ? 'ENABLED' : 'DISABLED'}`);
});
