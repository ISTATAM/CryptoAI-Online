import { DurableObject } from 'cloudflare:workers';

const json = (value, status = 200) =>
  Response.json(value, { status });

const splitIDs = value =>
  [...new Set(String(value || '')
    .split(',')
    .map(x => x.trim())
    .filter(Boolean))];

const admins = env => splitIDs(env.ADMIN_LINE_USER_ID);

const recipients = env =>
  splitIDs(env.NOTIFY_LINE_USER_IDS || env.ADMIN_LINE_USER_ID);

const time = value => {
  const date = new Date(value);
  if (!value || !Number.isFinite(date.getTime())) return '未知';

  return new Intl.DateTimeFormat('zh-TW', {
    timeZone: 'Asia/Taipei',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  }).format(date);
};

const num = value =>
  value != null && Number.isFinite(Number(value))
    ? Number(value).toFixed(2)
    : '—';

const pct = value => `${num(value)}%`;

function validate(data) {
  if (!data || !Array.isArray(data.top_predictions)) {
    throw new Error('缺少 top_predictions 陣列');
  }

  const date = new Date(data.recorded_at_utc);
  if (!data.recorded_at_utc || !Number.isFinite(date.getTime())) {
    throw new Error('recorded_at_utc 格式錯誤');
  }

  const threshold = Number(data.threshold ?? 0.7);
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error('threshold 必須為 0～1');
  }

  const list = data.top_predictions.map(item => {
    const score = Number(item.ai_score);

    if (
      !/^[A-Z0-9]{2,30}USDT$/.test(item.symbol || '') ||
      item.ai_score == null ||
      !Number.isFinite(score) ||
      score < 0 ||
      score > 1
    ) {
      throw new Error('symbol 或 ai_score 格式錯誤');
    }

    return { ...item, ai_score: score };
  }).sort((a, b) => b.ai_score - a.ai_score);

  return { ...data, threshold, top_predictions: list };
}

async function digest(value) {
  const bytes = new Uint8Array(
    await crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode(value)
    )
  );

  return [...bytes]
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

function equal(a, b) {
  if (
    typeof a !== 'string' ||
    typeof b !== 'string' ||
    a.length !== b.length
  ) return false;

  let difference = 0;
  for (let i = 0; i < a.length; i++) {
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return difference === 0;
}

async function signed(raw, signature, secret) {
  if (!signature || !secret) return false;

  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );

  const bytes = new Uint8Array(
    await crypto.subtle.sign(
      'HMAC',
      key,
      new TextEncoder().encode(raw)
    )
  );

  return equal(
    btoa(String.fromCharCode(...bytes)),
    signature
  );
}

async function gh(env, path, options = {}) {
  const owner = encodeURIComponent(env.GH_OWNER);
  const repo = encodeURIComponent(env.GH_REPO);

  const response = await fetch(
    `https://api.github.com/repos/${owner}/${repo}/${path}`,
    {
      ...options,
      headers: {
        Accept: 'application/vnd.github+json',
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'CryptoAI-V10',
        'Content-Type': 'application/json',
        ...options.headers
      },
      signal: AbortSignal.timeout(15000)
    }
  );

  if (!response.ok) {
    const reason =
      response.status === 401 ? '請檢查 GITHUB_TOKEN' :
      response.status === 403 ? '請檢查權限或 API 額度' :
      response.status === 404 ? '請檢查儲存庫、分支及檔案路徑' :
      '請稍後重試';

    throw new Error(`GitHub HTTP ${response.status}：${reason}`);
  }

  return response;
}

async function getData(env, path) {
  const encodedPath = path.split('/').map(encodeURIComponent).join('/');
  const branch = encodeURIComponent(env.GH_BRANCH || 'main');

  const response = await gh(
    env,
    `contents/${encodedPath}?ref=${branch}`,
    { headers: { Accept: 'application/vnd.github.raw+json' } }
  );

  return response.json();
}

async function line(env, endpoint, payload, retryKey) {
  const response = await fetch(
    `https://api.line.me/v2/bot/message/${endpoint}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${env.LINE_CHANNEL_ACCESS_TOKEN}`,
        ...(retryKey ? { 'X-Line-Retry-Key': retryKey } : {})
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(15000)
    }
  );

  if (
    response.ok ||
    (
      retryKey &&
      response.status === 409 &&
      response.headers.get('x-line-accepted-request-id')
    )
  ) return;

  throw new Error(`LINE HTTP ${response.status}`);
}

function today(data) {
  const list = data.top_predictions.filter(
    item => item.ai_score >= data.threshold
  );

  return [
    '📈 CryptoAI V10 今日訊號',
    '',
    `最近掃描：${time(data.recorded_at_utc)}`,
    `成功掃描：${data.successful_scans ?? '未知'} 幣`,
    `門檻：${pct(data.threshold * 100)}`,
    '',
    list.length
      ? list.slice(0, 30).map((item, index) =>
          `${index + 1}. ${item.symbol}｜AI 分數 ${pct(item.ai_score * 100)}`
        ).join('\n')
      : '目前沒有符合門檻的訊號。',
    ...(list.length > 30 ? ['僅顯示前 30 筆。'] : []),
    '',
    'AI 分數並非經校準的獲利機率；目前為模擬測試。'
  ].join('\n');
}

function performance(data) {
  const key =
    data.performance_by_cost?.['0.3%'] ? '0.3%' :
    data.performance_by_cost?.['0.2%'] ? '0.2%' :
    null;

  const result = key ? data.performance_by_cost[key] : {};
  const counts = data.status_counts || {};

  return [
    '💹 CryptoAI Forward Test',
    `更新：${time(data.updated_at_utc)}`,
    '',
    `總訊號：${data.total_signals ?? '未知'}`,
    `已完成：${data.closed_trades ?? '未知'}`,
    `等待進場：${counts.WAIT_ENTRY ?? 0}`,
    `等待出場：${counts.WAIT_EXIT ?? 0}`,
    '',
    `成本情境：${key ?? '無資料'}`,
    `初始資金：${num(data.rules?.initial_capital_usdt ?? 100)} USDT`,
    `目前資金：${num(result.final_capital_usdt)} USDT`,
    `總報酬：${pct(result.total_return_pct)}`,
    `勝率：${pct(result.win_rate_pct)}`,
    `最大回撤：${pct(result.max_drawdown_pct)}`,
    `平均單筆：${pct(result.average_trade_return_pct)}`,
    '',
    '此為模擬績效，尚非真實下單。'
  ].join('\n');
}

const help = [
  '📖 CryptoAI V10',
  '',
  '狀態：檢查 GitHub 最新資料',
  '今日訊號：符合門檻的最新訊號',
  '前10名：AI 排名',
  '績效：模擬交易績效',
  'BTC／ETH／BTCUSDT：查詢幣種',
  '我的ID：查看自己的 LINE ID',
  '說明：顯示本清單',
  '',
  '以下限管理者：',
  '啟動AI：要求 GitHub 啟動掃描',
  '新增使用者 U完整ID 名稱',
  '移除使用者 U完整ID',
  '使用者清單'
].join('\n');

async function access(env, actor, text = '') {
  if (!env.NOTIFY_STATE) throw new Error('未設定 NOTIFY_STATE');

  const object = env.NOTIFY_STATE.get(
    env.NOTIFY_STATE.idFromName('cryptoai-v10')
  );

  const response = await object.fetch('https://internal/access', {
    method: 'POST',
    body: JSON.stringify({ actor, text })
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '權限服務失敗');
  return data;
}

async function command(text, env, role) {
  if (text === '說明' || text.toLowerCase() === 'help') return help;

  if (text === '啟動AI') {
    if (role !== 'admin') return '⛔ 只有管理者可以啟動 AI。';

    const workflow = encodeURIComponent(
      env.GH_WORKFLOW || 'v9_paper_trade.yml'
    );

    await gh(env, `actions/workflows/${workflow}/dispatches`, {
      method: 'POST',
      body: JSON.stringify({ ref: env.GH_BRANCH || 'main' })
    });

    return [
      '🚀 GitHub 已接受 AI 掃描請求。',
      '',
      '稍後請到 Actions 確認執行結果。',
      '掃描及通知成功後會收到 LINE。',
      '這則回覆不代表掃描已完成。'
    ].join('\n');
  }

  if (text === '績效') {
    return performance(
      await getData(env, 'paper_trading/performance_summary.json')
    );
  }

  const symbol = text.toUpperCase();
  if (
    !['狀態', '今日訊號', '前10名'].includes(text) &&
    !/^[A-Z0-9]{2,30}$/.test(symbol)
  ) return help;

  const data = validate(
    await getData(env, 'paper_trading/latest_summary.json')
  );

  if (text === '狀態') {
    const age = (
      Date.now() - new Date(data.recorded_at_utc).getTime()
    ) / 3600000;

    const permission = await access(env, '');

    return [
      '🟢 LINE 指令與 GitHub 資料讀取正常',
      '',
      `最新掃描：${time(data.recorded_at_utc)}`,
      `資料距今：${age.toFixed(1)} 小時${age > 30 ? '（資料較舊，請檢查 Actions）' : ''}`,
      `成功掃描：${data.successful_scans ?? '未知'} 幣`,
      `通知對象：${permission.count} 位`,
      '',
      '此檢查不代表主動推送已成功。',
      '輸入「說明」查看指令。'
    ].join('\n');
  }

  if (text === '今日訊號') return today(data);

  if (text === '前10名') {
    return [
      '🏆 AI 前10名',
      `掃描：${time(data.recorded_at_utc)}`,
      '',
      data.top_predictions.slice(0, 10).map((item, index) =>
        `${index + 1}. ${item.symbol}｜${pct(item.ai_score * 100)}`
      ).join('\n')
    ].join('\n');
  }

  const target = symbol.endsWith('USDT') ? symbol : `${symbol}USDT`;
  const index = data.top_predictions.findIndex(
    item => item.symbol === target
  );

  if (index < 0) {
    return [
      `🔎 ${target}`,
      '',
      '不在 latest_summary.json 的 top_predictions 中。',
      '可能未掃描或未列入輸出，無法從這份資料判斷分數。',
      `掃描：${time(data.recorded_at_utc)}`
    ].join('\n');
  }

  const item = data.top_predictions[index];

  return [
    `🪙 ${target}`,
    `AI 分數：${pct(item.ai_score * 100)}`,
    `輸出清單排名：#${index + 1}`,
    `訊號門檻：${pct(data.threshold * 100)}`,
    `判斷：${item.ai_score >= data.threshold ? '已達訊號門檻' : '尚未達門檻'}`,
    `掃描：${time(data.recorded_at_utc)}`,
    '',
    'AI 分數不等於保證上漲機率。'
  ].join('\n');
}

export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;

    if (request.method === 'GET' && path === '/') {
      return new Response('CryptoAI V10 LINE Bot is running.');
    }

    if (path === '/notify') {
      if (request.method !== 'POST') {
        return new Response('Method Not Allowed', { status: 405 });
      }

      if (
        !env.NOTIFY_SECRET ||
        !equal(
          request.headers.get('Authorization'),
          `Bearer ${env.NOTIFY_SECRET}`
        )
      ) {
        return new Response('Unauthorized', { status: 401 });
      }

      if (!env.NOTIFY_STATE) {
        return json({ error: '未設定 NOTIFY_STATE 綁定' }, 500);
      }

      const raw = await request.text();
      if (raw.length > 1024 * 1024) {
        return json({ error: 'Payload too large' }, 413);
      }

      try {
        const body = JSON.parse(raw);
        const data = validate(body.summary);
        const age = Date.now() -
          new Date(data.recorded_at_utc).getTime();

        if (age > 36 * 3600000 || age < -10 * 60000) {
          return json({
            error: '掃描資料超過 36 小時或時間在未來'
          }, 422);
        }

        const object = env.NOTIFY_STATE.get(
          env.NOTIFY_STATE.idFromName('cryptoai-v10')
        );

        return await object.fetch('https://internal/notify', {
          method: 'POST',
          body: JSON.stringify({ summary: data })
        });
      } catch (error) {
        console.error('notify validation:', error.message);
        return json({ error: error.message }, 400);
      }
    }

    if (path !== '/' && path !== '/webhook') {
      return new Response('Not Found', { status: 404 });
    }

    if (request.method !== 'POST') {
      return new Response('Method Not Allowed', { status: 405 });
    }

    const raw = await request.text();

    if (!await signed(
      raw,
      request.headers.get('x-line-signature'),
      env.LINE_CHANNEL_SECRET
    )) {
      return new Response('Invalid signature', { status: 401 });
    }

    let body;
    try {
      body = JSON.parse(raw);
    } catch {
      return new Response('Invalid JSON', { status: 400 });
    }

    let failed = false;

    for (const event of body.events || []) {
      if (
        event.type !== 'message' ||
        event.message?.type !== 'text' ||
        !event.replyToken
      ) continue;

      // 避免 LINE 重送事件再次觸發 AI 掃描。
      if (event.deliveryContext?.isRedelivery) continue;

      let message = '⛔ 無權限使用此機器人。';

      try {
        const actor = event.source?.userId;
        const text = event.message.text.trim();

        if (event.source?.type !== 'user') {
          message = '請在與機器人的一對一聊天室輸入指令。';
        } else if (text === '我的ID') {
          message = `你的 LINE ID：\n${actor || '無法取得'}`;
        } else {
          const permission = await access(env, actor, text);

          if (permission.message) {
            message = permission.message;
          } else if (permission.role) {
            message = await command(text, env, permission.role);
          }
        }
      } catch (error) {
        console.error('command:', error.message);
        message = `⚠️ ${error.message}\n請檢查設定或稍後重試。`;
      }

      try {
        await line(env, 'reply', {
          replyToken: event.replyToken,
          messages: [{ type: 'text', text: message.slice(0, 4900) }]
        });
      } catch (error) {
        console.error('reply:', error.message);
        failed = true;
      }
    }

    return new Response(failed ? 'Reply failed' : 'OK', {
      status: failed ? 502 : 200
    });
  }
};

export class NotificationState extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.ctx = ctx;
    this.env = env;
    this.tail = Promise.resolve();
  }

  async fetch(request) {
    const body = await request.json();
    const path = new URL(request.url).pathname;

    const task = this.tail.then(() =>
      path === '/access'
        ? this.manage(body)
        : this.process(body.summary)
    );

    this.tail = task.catch(() => {});

    try {
      return json(await task);
    } catch (error) {
      console.error('state:', error.message);
      return json({ error: error.message }, 503);
    }
  }

  async users() {
    const stored = await this.ctx.storage.get('users') || {};

    return [...new Set([
      ...recipients(this.env),
      ...admins(this.env),
      ...Object.keys(stored)
    ])];
  }

  async manage({ actor, text = '' }) {
    const admin = admins(this.env).includes(actor);
    const stored = await this.ctx.storage.get('users') || {};
    const role = admin ? 'admin' : stored[actor] ? 'user' : null;
    const count = (await this.users()).length;

    if (!/^(新增使用者|移除使用者|使用者清單)/.test(text)) {
      return { role, count };
    }

    if (!admin) {
      return {
        role,
        count,
        message: '⛔ 只有管理者可以管理使用者。'
      };
    }

    if (text === '使用者清單') {
      const rows = (await this.users()).map(id => {
        const label = admins(this.env).includes(id)
          ? '管理者'
          : stored[id]?.name || '通知對象';

        return `${label}：${id}`;
      });

      return {
        role,
        count,
        message: `📋 使用者清單（${count} 位）\n\n${rows.join('\n')}`
      };
    }

    const match = text.match(
      /^(新增使用者|移除使用者)\s+(U[a-fA-F0-9]{32})(?:\s+(.{1,30}))?$/
    );

    if (!match) {
      return {
        role,
        count,
        message: [
          '格式：',
          '新增使用者 U開頭的完整ID 名稱',
          '移除使用者 U開頭的完整ID',
          '',
          '請對方先加好友，傳送「我的ID」。'
        ].join('\n')
      };
    }

    const [, action, id, name] = match;

    if (admins(this.env).includes(id)) {
      return {
        role,
        count,
        message: '這是設定中的管理者，無須新增，也不能用此指令移除。'
      };
    }

    if (action === '新增使用者') {
      if (!stored[id] && count >= 100) {
        return {
          role,
          count,
          message: '目前最多支援 100 位通知對象。'
        };
      }

      stored[id] = { name: name || '一般使用者' };
      await this.ctx.storage.put('users', stored);

      return {
        role,
        message: [
          `✅ 已新增 ${stored[id].name}`,
          id,
          '可以查詢資料，並接收後續通知。'
        ].join('\n')
      };
    }

    if (recipients(this.env).includes(id)) {
      return {
        role,
        count,
        message: '此 ID 設定在 NOTIFY_LINE_USER_IDS，請先從 Cloudflare 該變數移除，再輸入本指令。'
      };
    }

    if (!stored[id]) {
      return {
        role,
        count,
        message: '這個 ID 不在使用者清單。'
      };
    }

    delete stored[id];
    const pendingKey = `pending:${await digest(id)}`;

    await this.ctx.storage.transaction(async transaction => {
      await transaction.put('users', stored);
      await transaction.delete(pendingKey);
    });

    return {
      role,
      message: [
        '✅ 已移除使用者',
        id,
        '已停止後續通知與查詢權限。'
      ].join('\n')
    };
  }

  async sendPending(user, key) {
    const pending = await this.ctx.storage.get(key);
    if (!pending) return false;

    if (Date.now() - pending.created > 23 * 3600000) {
      throw new Error(
        '待送訊息超過 23 小時，需檢查 LINE 與通知紀錄，暫停自動重送'
      );
    }

    await line(this.env, 'push', {
      to: user,
      messages: pending.messages
    }, pending.retryKey);

    await this.ctx.storage.transaction(async transaction => {
      for (const id of pending.ids) {
        await transaction.put(id, Date.now());
      }
      await transaction.delete(key);
    });

    return true;
  }

  async process(data) {
    const users = await this.users();
    if (!users.length) throw new Error('未設定通知對象');

    const day = new Intl.DateTimeFormat('en-CA', {
      timeZone: 'Asia/Taipei',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit'
    }).format(new Date(data.recorded_at_utc));

    const signals = data.top_predictions.filter(
      item => item.ai_score >= data.threshold
    );

    const unique = new Map();

    for (const item of signals) {
      const date = new Date(item.signal_open_time);

      if (
        !item.signal_open_time ||
        !Number.isFinite(date.getTime())
      ) {
        throw new Error(
          `${item.symbol} 缺少有效 signal_open_time，無法安全去重`
        );
      }

      unique.set(
        `${item.symbol}|${date.toISOString()}`,
        item
      );
    }

    let pushed = 0;
    let skipped = 0;

    for (const user of users) {
      const pendingKey = `pending:${await digest(user)}`;

      if (await this.sendPending(user, pendingKey)) pushed++;

      const dailyId = `done:${await digest(`${user}|daily|${day}`)}`;
      const ids = [];
      const fresh = [];

      for (const [identity, item] of unique) {
        const id = `done:${await digest(`${user}|signal|${identity}`)}`;

        if (!await this.ctx.storage.get(id)) {
          fresh.push(item);
          ids.push(id);
        }
      }

      const daily =
        this.env.NOTIFY_DAILY_SUMMARY !== 'false' &&
        !await this.ctx.storage.get(dailyId);

      if (!daily && !fresh.length) {
        skipped++;
        continue;
      }

      let first = true;

      do {
        const chunk = fresh.splice(0, 25);
        const chunkIDs = ids.splice(0, 25);

        const rows = [
          chunk.length
            ? '🚨 CryptoAI V10 新訊號'
            : '📊 CryptoAI V10 每日摘要',
          '',
          `掃描：${time(data.recorded_at_utc)}`,
          `成功掃描：${data.successful_scans ?? '未知'} 幣`,
          `門檻：${pct(data.threshold * 100)}`,
          `本次輸出符合門檻：${signals.length} 筆`,
          ''
        ];

        if (chunk.length) {
          rows.push(...chunk.map(item =>
            `${item.symbol}｜AI ${pct(item.ai_score * 100)}\nK線：${time(item.signal_open_time)}`
          ));
        } else {
          rows.push('沒有尚未通知的新訊號。');
        }

        rows.push(
          '',
          '目前為模擬測試，AI 分數不等於獲利機率。'
        );

        if (first && daily) chunkIDs.push(dailyId);

        await this.ctx.storage.put(pendingKey, {
          created: Date.now(),
          retryKey: crypto.randomUUID(),
          ids: chunkIDs,
          messages: [{
            type: 'text',
            text: rows.join('\n').slice(0, 4900)
          }]
        });

        await this.sendPending(user, pendingKey);
        pushed++;
        first = false;
      } while (fresh.length);
    }

    return {
      ok: true,
      pushed_batches: pushed,
      skipped_recipients: skipped,
      scan_time: data.recorded_at_utc
    };
  }
}
