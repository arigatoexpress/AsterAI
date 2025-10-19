"use client";
import { gql, useQuery } from '@apollo/client';
import { ApolloProvider } from '@apollo/client';
import client from '@/lib/apolloClient';

const INSIGHTS = gql`
	query Insights { insights { symbol signal confidence reason } }
`;

function InsightsInner() {
	const { data, loading, error } = useQuery(INSIGHTS, { pollInterval: 4000 });
	if (loading) return <div style={{padding:24}}>Loading…</div>;
	if (error) return <div style={{padding:24,color:'#f66'}}>Error: {error.message}</div>;
	const insights = data?.insights ?? [];
	return (
		<div style={{padding:24, display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(240px,1fr))', gap:16}}>
			{insights.map((i:any) => (
				<div key={i.symbol} style={{border:'1px solid #222', borderRadius:12, padding:16}}>
					<div style={{display:'flex',justifyContent:'space-between'}}>
						<strong>{i.symbol}</strong>
						<span style={{color:'#8A2BE2'}}>{(i.confidence*100).toFixed(0)}%</span>
					</div>
					<div style={{marginTop:8}}>Signal: {i.signal}</div>
					<div style={{opacity:0.8,marginTop:8,fontSize:12}}>{i.reason}</div>
				</div>
			))}
		</div>
	);
}

export default function InsightsPage() {
	return (
		<ApolloProvider client={client}>
			<InsightsInner />
		</ApolloProvider>
	);
}


