"use client";

import React from 'react';

type Props = { size?: number };

export default function InfinitySymbol({ size = 120 }: Props) {
	const s = String(size);
	return (
		<svg width={s} height={s} viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg" aria-label="AsterAI Infinity">
			<defs>
				<linearGradient id="iris" x1="0%" y1="0%" x2="100%" y2="0%">
					<stop offset="0%" stopColor="#8A2BE2" />
					<stop offset="50%" stopColor="#00FFFF" />
					<stop offset="100%" stopColor="#FF69B4" />
				</linearGradient>
				<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
					<feGaussianBlur stdDeviation="3.5" result="coloredBlur" />
					<feMerge>
						<feMergeNode in="coloredBlur" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>
			</defs>
			<path
				d="M 20 50 C 20 20, 80 20, 100 50 C 120 80, 180 80, 180 50 C 180 20, 120 20, 100 50 C 80 80, 20 80, 20 50 Z"
				fill="none"
				stroke="url(#iris)"
				strokeWidth="6"
				filter="url(#glow)"
			/>
		</svg>
	);
}


