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
const ENABLE_THINKING_MODE = false;  // true면 chat_template_kwargs.thinking 전달

// axios 에러에서 NIM이 돌려준 실제 에러 body를 뽑아내는 헬퍼.
// stream 요청이 실패하면 response.data가 stream이라 한 번 모아서 파싱해야 함.
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

function sendError(res, error) {
  const status = error.response?.status || 500;
  // nimError는 비동기지만 아래 호출부에서 await 해서 넘겨줌
  return { status };
}

async function handleError(res, error, label) {
  const status = error.response?.status || 500;
  const nimError = await extractNimError(error);
  console.error(`${label}:`, status, nimError || error.message);

  if (res.headersSent) {
    // 스트리밍 도중 에러면 그냥 연결 종료
    return res.end();
  }

  if (nimError) {
    // NIM은 보통 { error: { message, type, ... } } 형태로 내려줌
    return res.status(status).json(nimError);
  }
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
    thinking_mode: ENABLE_THINKING_MODE
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
    // 들어온 body를 통째로 넘기되, 필요한 것만 덮어씀
    const nimRequest = {
      ...req.body,
      temperature: req.body.temperature ?? 0.6,
      max_tokens: req.body.max_tokens ?? 9024,
      stream: !!req.body.stream,
      // chat_template_kwargs는 최상위 필드로 보내야 함 (extra_body 아님)
      ...(ENABLE_THINKING_MODE && {
        chat_template_kwargs: { thinking: true }
      })
    };

    const stream = nimRequest.stream;

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
              // reasoning_content는 OpenAI 스펙에 없으므로 항상 제거
              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // 비스트리밍: 필요한 부분만 수정해서 그대로 전달
      const data = response.data;
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
});
