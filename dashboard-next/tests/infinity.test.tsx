import { render } from '@testing-library/react';
import InfinitySymbol from '../components/InfinitySymbol';

test('renders InfinitySymbol without crashing', () => {
	const { container } = render(<InfinitySymbol size={64} />);
	expect(container.querySelector('svg')).toBeTruthy();
});


