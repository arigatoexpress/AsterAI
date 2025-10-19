import { render, screen } from '@testing-library/react';
import { ApolloProvider, InMemoryCache, ApolloClient } from '@apollo/client';
import MarketsPage from '../app/markets/page';

function renderWithApollo(ui: React.ReactElement) {
	const client = new ApolloClient({ uri: '/api/graphql', cache: new InMemoryCache() });
	return render(<ApolloProvider client={client}>{ui}</ApolloProvider>);
}

test('renders Markets heading', () => {
	renderWithApollo(<MarketsPage />);
	expect(screen.getByText(/Markets/i)).toBeInTheDocument();
});


