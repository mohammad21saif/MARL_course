
"""
	For ease of use, please lay out your grid in Euclidean-plane format and NOT
	in numpy-type format. For example, if an object needs to be placed in the
	3rd row and 7th column of the gridworld numpy matrix, enter its location in your
	layout dict as [7,3]. The codebase will take care of the matrix-indexing for you.
	For example, the above object will be queried as grid[3, 7] when placed into the
	grid.

	NOTE: the origin (0,0) is the top-left corner of the grid. The positive direction
	along the x-axis counts to the right and the positive direction along the y-axis

"""

LINEAR = {
	'FOUR_PLAYERS': {
		'WALLS': [
			[0, 4],
			[1, 4],
			[2, 4],
			[2, 5],
			
   			[4, 0],
			[4, 1],
			[4, 2],
			[5, 2],
		
			[7, 4],
			[8, 4],
			[9, 4],
			[7, 5],
   
			[5, 7],
			[5, 8],
			[5, 9],
			[4, 7]
		],


		'PLATES': [
			[8, 5],
			[4, 8],
			[5, 1],
			[1, 5]
		],

		'AGENTS': [
			[1, 1],
			[8, 1],
			[1, 8],
			[8, 8]
		],

		'GOAL': [
			[4, 7]
		]
	}
}
	