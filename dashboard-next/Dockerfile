FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm i

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN mkdir -p public
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=8080
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY package.json ./package.json
RUN npm i --omit=dev && adduser -D -u 1001 nextjs
USER nextjs
EXPOSE 8080
CMD ["npm","start"]


