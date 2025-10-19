import { GoogleAuth } from 'google-auth-library';

const LOG_LEVEL = (process.env.LOG_LEVEL || 'info').toLowerCase();

export async function POST(req: Request) {
	const body = await req.text();
	const gatewayUrl = process.env.GRAPHQL_URL;
	if (!gatewayUrl) {
		return new Response(JSON.stringify({ error: 'GRAPHQL_URL not set' }), { status: 500 });
	}
	try {
		const auth = new GoogleAuth();
		const client = await auth.getIdTokenClient(gatewayUrl);
		const started = Date.now();
		const res = await client.request({
			url: gatewayUrl,
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			data: body,
		});
		if (LOG_LEVEL === 'debug') {
			console.log(JSON.stringify({ level: 'debug', op: 'proxy', target: gatewayUrl, status: 200, latencyMs: Date.now() - started }));
		}
		return new Response(JSON.stringify(res.data), { status: 200, headers: { 'content-type': 'application/json' } });
	} catch (e: any) {
		if (LOG_LEVEL !== 'silent') {
			console.log(JSON.stringify({ level: 'error', op: 'proxy', target: gatewayUrl, error: e?.message || 'gateway_error' }));
		}
		return new Response(JSON.stringify({ error: e?.message || 'gateway_error' }), { status: 502 });
	}
}


