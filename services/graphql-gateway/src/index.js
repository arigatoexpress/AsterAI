import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { gql } from 'graphql-tag';
import fetch from 'node-fetch';
// simple in-memory cache for fast repeated reads
let marketsCache = { data: [], ts: 0 };
const MARKETS_TTL_MS = 2000; // 2s cache to smooth upstream hiccups
let tradesLog = [];

const typeDefs = gql`
	type Market { symbol: String!, price: Float }
	type Position { id: ID!, symbol: String!, qty: Float!, entryPrice: Float! }

	type Insight { symbol: String!, signal: String!, confidence: Float!, reason: String }

	input TradeInput { symbol: String!, side: String!, qty: Float!, price: Float!, ts: String }
type Trade { id: ID!, symbol: String!, side: String!, qty: Float!, price: Float!, ts: String!, pnl: Float, simpleExplanation: String, technicalExplanation: String }
	type PnlSummary { total: Float!, realized: Float!, unrealized: Float! }

	type Query {
		markets: [Market!]!
		positions: [Position!]!
		trades: [Trade!]!
		pnlSummary: PnlSummary!
		insights: [Insight!]!
	}

	type Mutation {
		recordTrade(input: TradeInput!): Trade!
	}
`;

const DATA_COLLECTOR_URL = process.env.DATA_COLLECTOR_URL; // e.g., https://collector-...a.run.app
const TRADER_URL = process.env.TRADER_URL; // e.g., https://trader-...a.run.app
const LOG_LEVEL = (process.env.LOG_LEVEL || 'info').toLowerCase();

async function safeJson(url, init) {
	const controller = new AbortController();
	const t = setTimeout(() => controller.abort(), 1500); // lower timeout for snappier UI
	try {
		const started = Date.now();
		const res = await fetch(url, { ...init, signal: controller.signal });
		const latencyMs = Date.now() - started;
		if (LOG_LEVEL === 'debug') {
			console.log(JSON.stringify({ level: 'debug', op: 'fetch', url, status: res.status, latencyMs }));
		}
		if (!res.ok) {
			throw new Error(`HTTP ${res.status}`);
		}
		return await res.json();
	} finally {
		clearTimeout(t);
	}
}

function generateExplanations(input) {
	const dir = input.side === 'BUY' ? 'long' : 'short';
	const simple = `Entered a ${dir} on ${input.symbol} at ${input.price}. Conditions suggested a favorable setup.`;
	const technical = `Signal favored ${dir}; entry price ${input.price}, qty ${input.qty}. Position sizing per risk budget; risk managed via adaptive SL/TP.`;
	return { simpleExplanation: simple, technicalExplanation: technical };
}

const resolvers = {
	Query: {
		markets: async () => {
			if (DATA_COLLECTOR_URL) {
				try {
					// serve cached data if fresh
					const now = Date.now();
					if (now - marketsCache.ts < MARKETS_TTL_MS && marketsCache.data?.length) {
						if (LOG_LEVEL === 'debug') {
							console.log(JSON.stringify({ level: 'debug', resolver: 'markets', cache: true, items: marketsCache.data.length }));
						}
						return marketsCache.data;
					}

					const upstream = `${DATA_COLLECTOR_URL}/api/markets`;
					const data = await safeJson(upstream, { headers: { 'accept': 'application/json' } });
					if (LOG_LEVEL === 'debug') {
						console.log(JSON.stringify({ level: 'debug', resolver: 'markets', upstream, items: Array.isArray(data) ? data.length : 0 }));
					}
					const mapped = Array.isArray(data) ? data.map((d) => ({ symbol: d.symbol, price: d.price })) : [];
					marketsCache = { data: mapped, ts: Date.now() };
					return mapped;
				} catch (e) {
					if (LOG_LEVEL !== 'silent') {
						console.log(JSON.stringify({ level: 'warn', resolver: 'markets', error: String(e) }));
					}
					// serve last known good if available to avoid UI stalls
					if (marketsCache.data?.length) {
						return marketsCache.data;
					}
					return [];
				}
			}
			// Fallback stub
			return [{ symbol: 'BTCUSDT', price: 50000 }];
		},
		positions: async () => {
			if (TRADER_URL) {
				try {
					const upstream = `${TRADER_URL}/api/positions`;
					const data = await safeJson(upstream, { headers: { 'accept': 'application/json' } });
					if (LOG_LEVEL === 'debug') {
						console.log(JSON.stringify({ level: 'debug', resolver: 'positions', upstream, items: Array.isArray(data) ? data.length : 0 }));
					}
					return Array.isArray(data) ? data.map((p) => ({ id: String(p.id ?? `${p.symbol}-${p.qty}`), symbol: p.symbol, qty: p.qty, entryPrice: p.entry_price ?? p.entryPrice })) : [];
				} catch (e) {
					if (LOG_LEVEL !== 'silent') {
						console.log(JSON.stringify({ level: 'warn', resolver: 'positions', error: String(e) }));
					}
					return [];
				}
			}
			return [];
		},
		trades: async () => tradesLog,
		pnlSummary: async () => {
			const realized = tradesLog.reduce((a, t) => a + (t.pnl || 0), 0);
			return { total: realized, realized, unrealized: 0 };
		},
		insights: async () => {
			// naive placeholder insights based on market cache
			const items = marketsCache.data?.slice(0,6) || [{ symbol:'BTCUSDT', price:50000 }];
			return items.map((m) => ({ symbol: m.symbol, signal: 'watch', confidence: 0.6, reason: 'Monitoring momentum and volatility' }));
		},
	},
  Mutation: {
    recordTrade: async (_root, { input }) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
      const ts = input.ts || new Date().toISOString();
      const expl = generateExplanations(input);
      const trade = { id, symbol: input.symbol, side: input.side, qty: input.qty, price: input.price, ts, pnl: 0, ...expl };
      tradesLog.push(trade);
      return trade;
    }
  }
};

async function start() {
	const app = express();
	app.disable('x-powered-by');
	app.use(cors({ origin: false })); // private; only server-side callers
	app.use((req, _res, next) => {
		const start = Date.now();
		req.on('end', () => {
			const ms = Date.now() - start;
			console.log(JSON.stringify({ level: 'info', service: 'graphql-gateway', method: req.method, path: req.path, latencyMs: ms }));
		});
		next();
	});
	app.use(bodyParser.json({ limit: '1mb' }));

	const server = new ApolloServer({ typeDefs, resolvers });
	await server.start();
	app.use('/graphql', expressMiddleware(server));

	app.get('/health', (_, res) => res.status(200).json({ status: 'ok' }));

	const port = process.env.PORT || 8080;
	app.listen(port, () => console.log(`GraphQL gateway listening on :${port}`));
}

start();


