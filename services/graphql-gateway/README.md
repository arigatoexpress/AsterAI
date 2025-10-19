# AsterAI GraphQL Gateway

Private Apollo Server that stitches upstream REST services (trader, data collector, metrics). Exposed only to Cloud Run IAM callers; the public dashboard talks to this via a server-side proxy.

## Env
- `PROJECT_ID` – GCP project id (optional)
- `DATA_COLLECTOR_URL` – base URL for market data service
- `TRADER_URL` – base URL for live trading service
- `TIMEOUT_MS` – upstream request timeout (default 3500)

## Build and deploy
```
 gcloud builds submit . --config cloudbuild.yaml --substitutions _TAG=prod

 gcloud run deploy aster-graphql-gateway \
   --image gcr.io/$(gcloud config get-value project)/aster-graphql-gateway:prod \
   --region us-central1 \
   --no-allow-unauthenticated \
   --set-env-vars PROJECT_ID=$(gcloud config get-value project) \
   --set-env-vars TIMEOUT_MS=3500 \
   --set-env-vars DATA_COLLECTOR_URL=https://<data-collector-url> \
   --set-env-vars TRADER_URL=https://<trader-url>
```

## Schema (excerpt)
- Query
  - `markets: [Market!]!`
  - `positions: [Position!]!`
- Types
  - `Market { symbol: String!, price: Float }`
  - `Position { id: ID!, symbol: String!, qty: Float!, entryPrice: Float! }`

## Security
- Cloud Run IAM only (no public access).
- CORS disabled for browsers; server-to-server only.
