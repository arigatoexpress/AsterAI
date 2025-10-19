/** @type {import('next').NextConfig} */
const nextConfig = {
	reactStrictMode: true,
	headers: async () => {
		const csp = [
			"default-src 'self'",
			"script-src 'self' 'strict-dynamic'",
			"style-src 'self' 'unsafe-inline'",
			"img-src 'self' data: blob:",
			"connect-src 'self' https:",
			"font-src 'self' data:",
			"object-src 'none'",
			"base-uri 'self'",
			"frame-ancestors 'none'"
		].join('; ');
		return [
			{
				source: '/(.*)',
				headers: [
					{ key: 'Content-Security-Policy', value: csp },
					{ key: 'Referrer-Policy', value: 'no-referrer' },
					{ key: 'X-Content-Type-Options', value: 'nosniff' },
					{ key: 'X-Frame-Options', value: 'DENY' },
					{ key: 'Permissions-Policy', value: 'geolocation=(), microphone=(), camera=()' },
					{ key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' }
				]
			}
		];
	}
};

export default nextConfig;


