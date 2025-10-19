# AsterAI Next.js Dashboard

Secure, SSR-first dashboard powered by Next.js 15 + TypeScript. Uses a server-side GraphQL proxy to call a private GraphQL Gateway on Cloud Run via IAM identity tokens. The browser never calls private services directly.

## Local development

- Node 20+
- Install deps: `npm i`
- Run dev: `npm run dev`
- Env (local):
  - `GRAPHQL_URL` = http://localhost:8081/graphql (when running gateway locally)

## Cloud Run deployment

Recommended service/region:
- Dashboard service: `aster-next-dashboard` (region: `us-central1`)
- GraphQL Gateway: `aster-graphql-gateway` (region: `us-central1`)

Build with Cloud Build:
```
 gcloud builds submit . --config cloudbuild.yaml --substitutions _TAG=prod
```

Deploy to Cloud Run:
```
 gcloud run deploy aster-next-dashboard \
   --image gcr.io/$(gcloud config get-value project)/aster-dashboard-next:prod \
   --region us-central1 \
   --allow-unauthenticated \
   --set-env-vars GRAPHQL_URL=https://aster-graphql-gateway-$(gcloud config get-value project-number)-uc.a.run.app/graphql
```

Notes:
- `GRAPHQL_URL` is the private gateway URL; the dashboard will mint an IAM id token on the server and forward requests.
- Grant the dashboard service account Run Invoker on the gateway:
```
 dashboard_sa=$(gcloud run services describe aster-next-dashboard --region us-central1 --format='value(template.spec.serviceAccountName)')
 gcloud run services add-iam-policy-binding aster-graphql-gateway \
   --region us-central1 \
   --member=serviceAccount:${dashboard_sa} \
   --role=roles/run.invoker
```

## Security
- Strict CSP set in `next.config.js` headers.
- No secrets in client bundle; server-only env used in `/api/graphql` proxy.
- Cloud Run IAM enforced on the gateway.

## Structure
- `app/` – RSC/SSR routes
- `app/api/graphql` – IAM proxy
- `components/InfinitySymbol.tsx` – brand visual
- `lib/apolloClient.ts` – client bound to `/api/graphql`
