"use client";
import { gql, useQuery } from '@apollo/client';
import { ApolloProvider } from '@apollo/client';
import client from '@/lib/apolloClient';

const JOURNAL = gql`
	query Journal {
		trades { id symbol side qty price ts pnl }
		pnlSummary { total realized unrealized }
	}
`;

function JournalInner() {
	const { data, loading, error } = useQuery(JOURNAL, { pollInterval: 5000 });
	if (loading) return <div style={{padding:24}}>Loading…</div>;
	if (error) return <div style={{padding:24,color:'#f66'}}>Error: {error.message}</div>;
	const trades = data?.trades ?? [];
	const pnl = data?.pnlSummary ?? { total: 0, realized: 0, unrealized: 0 };
	return (
		<div style={{padding:24}}>
			<h2 style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
				<span>Trade Journal</span>
				<span style={{fontSize:14,color:'#8A2BE2'}}>Total P&L: {pnl.total.toFixed(2)}</span>
			</h2>
			<table style={{width:'100%',borderCollapse:'collapse'}}>
				<thead>
					<tr>
						<th align="left">Time</th>
						<th align="left">Symbol</th>
						<th align="left">Side</th>
						<th align="right">Qty</th>
						<th align="right">Price</th>
						<th align="right">PnL</th>
					</tr>
				</thead>
				<tbody>
					{trades.map((t:any) => (
						<tr key={t.id} style={{borderTop:'1px solid #222'}}>
							<td>{new Date(t.ts).toLocaleString()}</td>
							<td>{t.symbol}</td>
							<td style={{color: t.side==='BUY'?'#00FFFF':'#FF69B4'}}>{t.side}</td>
							<td align="right">{t.qty}</td>
							<td align="right">{t.price}</td>
							<td align="right" style={{color:(t.pnl||0)>=0?'#00C853':'#FF5252'}}>{(t.pnl||0).toFixed(2)}</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
}

export default function JournalPage() {
	return (
		<ApolloProvider client={client}>
			<JournalInner />
		</ApolloProvider>
	);
}


