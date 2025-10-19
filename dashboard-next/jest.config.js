/** @type {import('jest').Config} */
export default {
	testEnvironment: 'jsdom',
	roots: ['<rootDir>'],
	transform: { '^.+\\.(ts|tsx)$': ['ts-jest', { tsconfig: '<rootDir>/tsconfig.json' }] },
	setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
	moduleNameMapper: {
		'^@/(.*)$': '<rootDir>/$1'
	}
};


