use foldhash::HashSet;
use graphidx::{data::MatrixDataSource, graphs::{Graph, WeightedGraph}, heaps::MinHeap, indices::{GraphIndex, GreedyLayeredGraphIndex, GreedySingleGraphIndex}, measures::Distance, types::{SyncFloat, SyncUnsignedInteger}};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::hnsw::{HNSWParallelHeapBuilder, HNSWParams, HNSWStyleBuilder};

struct ObservedEdgesStore<R: SyncUnsignedInteger> {
	observed_edges: Vec<HashSet<R>>,
}
impl<R: SyncUnsignedInteger> ObservedEdgesStore<R> {
	fn new(n_elements: usize) -> Self {
		Self { observed_edges: vec![HashSet::default(); n_elements] }
	}
	fn observe_edge(&mut self, i: R, j: R) -> bool {
		if i != j {
			let (i,j) = (i.min(j), i.max(j));
			unsafe{self.observed_edges.get_unchecked_mut(i.to_usize().unwrap_unchecked()).insert(j)}
		} else { false }
	}
}
struct UnionFind<R: SyncUnsignedInteger> {
	parents: Vec<R>,
}
impl<R: SyncUnsignedInteger> UnionFind<R> {
	fn new(n_elements: usize) -> Self {
		assert!(R::max_value().to_usize().unwrap() >= n_elements);
		assert!(isize::max_value() as usize >= n_elements);
		let mut parents = Vec::with_capacity(n_elements);
		unsafe{parents.extend((0..n_elements).map(|i| R::from(i).unwrap_unchecked()));}
		Self { parents }
	}
	fn find_immutable(&self, i: R) -> R {
		unsafe {
			let parents = self.parents.as_ptr();
			let mut i = i;
			let mut par = parents.offset(i.to_isize().unwrap());
			while *par != i {
				(i, par) = (*par, parents.offset(i.to_isize().unwrap()));
			}
			i
		}
	}
	/* Gets the root node of the union find and returns it */
	fn find(&mut self, i: R) -> R {
		unsafe {
			let parents = self.parents.as_mut_ptr();
			let mut i = i;
			let mut par = parents.offset(i.to_isize().unwrap());
			while *par != i {
				/* Path splitting */
				let next_par = parents.offset((*par).to_isize().unwrap());
				(i, *par, par) = (*par, *next_par, next_par);
			}
			i
		}
	}
	/* Combines two sets by making the root of the first set the child
	 * of the second set */
	fn union(&mut self, i: R, j: R) {
		let i_root = self.find(i);
		let j_root = self.find(j);
		self.parents[i_root.to_usize().unwrap()] = j_root;
	}
}

pub fn hnsw_based_dendrogram<
	F: SyncFloat,
	R: SyncUnsignedInteger,
	M: MatrixDataSource<F>+graphidx::types::Sync,
	Dist: Distance<F>+Sync+Send,
>(data: &M, dist: Dist, min_pts: usize, expand: bool, symmetric_expand: bool, hnsw_params: HNSWParams<F>) -> (Vec<(usize, usize, F, usize)>, Vec<F>) {
	let n = data.n_rows();
	assert!(R::max_value().to_usize().unwrap() >= n);
	/* Build HNSW on the data and get the graphs */
	let hnsw = HNSWParallelHeapBuilder::<R,_,_>::base_init(data, dist, hnsw_params);
	let (graphs, _, global_ids, dist) = hnsw._into_parts();
	/* Call generic function */
	graph_based_dendrogram(data, graphs, global_ids, dist, min_pts, expand, symmetric_expand)
}

pub fn hnsw_based_dendrogram_self_joined<
	F: SyncFloat,
	R: SyncUnsignedInteger,
	M: MatrixDataSource<F>+graphidx::types::Sync,
	Dist: Distance<F>+Sync+Send,
	>(data: &M, dist: Dist, min_pts: usize, expand: bool, symmetric_expand: bool, hnsw_params: HNSWParams<F>, self_join_neighbors: usize, query_max_heap_size: usize, query_local: bool) -> (Vec<(usize, usize, F, usize)>, Vec<F>) {
	let n = data.n_rows();
	assert!(R::max_value().to_usize().unwrap() >= n);
	/* Build HNSW on the data and get the graphs */
	let hnsw = HNSWParallelHeapBuilder::<R,_,_>::base_init(data, dist, hnsw_params);
	let (mut graphs, local_ids, global_ids, dist) = hnsw._into_parts();
	let graphs = if query_local {
		let index = GreedySingleGraphIndex::new(data, graphs.remove(0), dist.clone(), None);
		let self_join = index.self_join_query_local(self_join_neighbors, query_max_heap_size);
		let mut new_graphs: Vec<_> = Vec::new();
		new_graphs.push(self_join);
		graphs.into_iter().for_each(|g| new_graphs.push(g.as_weighted_dir_lol_graph()));
		new_graphs
	} else {
		let index = GreedyLayeredGraphIndex::new(data, graphs, local_ids, global_ids.clone(), dist.clone(), 1, None);
		let self_join = index.self_join_query(self_join_neighbors, query_max_heap_size);
		let mut new_graphs: Vec<_> = Vec::new();
		new_graphs.push(self_join);
		index.graphs()[1..].into_iter().for_each(|g| new_graphs.push(g.as_weighted_dir_lol_graph()));
		new_graphs
	};
	/* Call generic function */
	graph_based_dendrogram(data, graphs, global_ids, dist, min_pts, expand, symmetric_expand)
}

pub fn graph_based_dendrogram<
	F: SyncFloat,
	R: SyncUnsignedInteger,
	M: MatrixDataSource<F>+graphidx::types::Sync,
	Dist: Distance<F>+Sync+Send,
	G: WeightedGraph<R,F>,
>(data: &M, graphs: Vec<G>, global_ids: Vec<Vec<R>>, dist: Dist, min_pts: usize, expand: bool, symmetric_expand: bool) -> (Vec<(usize, usize, F, usize)>, Vec<F>) {
	let n = data.n_rows();
	assert!(R::max_value().to_usize().unwrap() >= n);
	/* Create storage for observed edges and accessor function */
	let mut observed_edges = ObservedEdgesStore::new(n);
	/* Create an expand queue for edges and insert all edges from all graph levels */
	let mut expand_queue = MinHeap::with_capacity(graphs.iter().map(|g| g.n_edges()).sum::<usize>() + 1_000);
	(0..graphs.len()).for_each(|i| {
		let graph = &graphs[i];
		if i > 0 { /* Higher layers */
			let id_map = global_ids.get(i-1).unwrap();
			(0..graph.n_vertices()).for_each(|i_node| {
				let i_global = *unsafe{id_map.get_unchecked(i_node)};
				let i_node = unsafe{R::from(i_node).unwrap_unchecked()};
				graph.foreach_neighbor_with_zipped_weight(i_node, |&d, &j_node| {
					let j_global = *unsafe{id_map.get_unchecked(j_node.to_usize().unwrap_unchecked())};
					if observed_edges.observe_edge(i_global, j_global) {
						expand_queue.push(d, (i_global.min(j_global), i_global.max(j_global)));
					}
				});
			});
		} else { /* Bottom layer */
			(0..graph.n_vertices()).for_each(|i_node| {
				let i_node = unsafe{R::from(i_node).unwrap_unchecked()};
				graph.foreach_neighbor_with_zipped_weight(i_node, |&d, &j_node| {
					if observed_edges.observe_edge(i_node, j_node) {
						expand_queue.push(d, (i_node.min(j_node), i_node.max(j_node)));
					}
				});
			});
		}
	});
	/* Get bottom layer graph for all further operations
	 * and translate into a more efficient condensed format */
	let graph = graphs.get(0).unwrap().as_dir_lol_graph();
	// let graph = graphs.get(0).unwrap().to_owned();
	// let start_time = std::time::Instant::now();
	/* Create a union find structure for the clusters */
	let mut union_find: UnionFind<R> = UnionFind::new(n);
	/* Create a storage for the dendrogram cluster ids */
	let mut cluster_ids = (0..n).collect::<Vec<usize>>();
	/* Create a storage for the cluster sizes */
	let mut cluster_sizes = vec![1; n];
	/* Create a storage for the dendrogram */
	let mut dendrogram = Vec::with_capacity(n - 1);
	/* Create a storage to count the number of visited edges for each node
	 * and a storage for the distance at which any point becomes a core point */
	let neighbor_counts = vec![0usize; n];
	let mut neighbor_stacks = (0..n).map(|_| MinHeap::<F,R>::with_capacity(min_pts-1)).collect::<Vec<_>>();
	let mut core_distances = vec![-F::one(); n];
	let mut core_point_count = 0;
	/* Do the actual clustering */
	unsafe {
		/* Shorthand to merge two clusters */
		let merge_clusters = |
			dendrogram: &mut Vec<(usize,usize,F,usize)>,
			union_find: &mut UnionFind<R>,
			cluster_ids: &mut Vec<usize>,
			cluster_sizes: &mut Vec<usize>,
			distance: F, i_root: usize, j_root: usize
		| {
			/* Update dendrogram info */
			let i_cluster_id = cluster_ids.get_unchecked(i_root);
			let j_cluster_id = cluster_ids.get_unchecked(j_root);
			let i_cluster_size = cluster_sizes.get_unchecked(i_root);
			let j_cluster_size = cluster_sizes.get_unchecked(j_root);
			let new_id = n + dendrogram.len();
			let new_size = *i_cluster_size + *j_cluster_size;
			dendrogram.push((*i_cluster_id, *j_cluster_id, distance, new_size));
			/* Update union find and cluster infos */
			union_find.union(R::from(i_root).unwrap_unchecked(), R::from(j_root).unwrap_unchecked());
			let new_root = j_root;
			*cluster_ids.get_unchecked_mut(new_root) = new_id;
			*cluster_sizes.get_unchecked_mut(new_root) = new_size;
		};
		while dendrogram.len() < n-1 && expand_queue.size() > 0 {
			/* Get the next edge */
			let (d_ij, (i, j)) = expand_queue.pop().unwrap_unchecked();
			let i_usize = i.to_usize().unwrap_unchecked();
			let j_usize = j.to_usize().unwrap_unchecked();
			/* Update core point info */
			let both_core_points = vec![(i, i_usize, j), (j, j_usize, i)].iter().map(|&(idx, idx_usize, other_idx)| {
				let cnt = neighbor_counts.as_ptr().offset(idx_usize as isize) as *mut usize;
				*cnt += 1;
				if *cnt==min_pts {
					*core_distances.get_unchecked_mut(idx_usize) = d_ij;
					core_point_count += 1;
					/* Get root objects */
					let root = union_find.find(idx).to_usize().unwrap_unchecked();
					/* Attempt to merge with all neighbors first */
					let neighbor_stack = neighbor_stacks.get_unchecked_mut(idx_usize);
					while neighbor_stack.size() > 0 {
						let (_distance,other) = neighbor_stack.pop().unwrap_unchecked();
						let other_usize = other.to_usize().unwrap_unchecked();
						if *neighbor_counts.get_unchecked(other_usize) >= min_pts {
							let other_root = union_find.find(other).to_usize().unwrap_unchecked();
							/* The core distance of this point should be larger than all distances seen so far, so just use d_ij. */
							if root != other_root {
								merge_clusters(
									&mut dendrogram,
									&mut union_find,
									&mut cluster_ids,
									&mut cluster_sizes,
									d_ij, other_root, root
								);
							}
						}
					}
				} else if *cnt < min_pts {
					neighbor_stacks.get_unchecked_mut(idx_usize).push(d_ij,other_idx);
				}
				*cnt >= min_pts
			}).all(|b| b);
			/* Merge clusters if both are core points now */
			if both_core_points {
				/* Get root objects again, in case they changed due to ops before */
				let i_root = union_find.find(i).to_usize().unwrap_unchecked();
				let j_root = union_find.find(j).to_usize().unwrap_unchecked();
				if i_root == j_root { continue; }
				/* Update dendrogram info */
				merge_clusters(
					&mut dendrogram,
					&mut union_find,
					&mut cluster_ids,
					&mut cluster_sizes,
					/* When i and j are core-points, d_ij should be smaller than the core distance, so no need to take any max */
					d_ij, i_root, j_root
				);
			}
			/* Expand on the edge and add new entries to the expand queue
			 * unless that is disabled with `expand = false` */
			if expand {
				/* Compute pairwise distances between neighborhoods in parallel.
				* Skip loops and already merged pairs. */
				let work_output = if symmetric_expand {
					/* Extend the expand queue with neighbor-of-neighbor pairs */
					let nodes1: Vec<R> = graph.iter_neighbors(i).cloned().chain(std::iter::once(i)).collect();
					let nodes2: Vec<R> = graph.iter_neighbors(j).cloned().chain(std::iter::once(j)).collect();
					let total_work = nodes1.len() * nodes2.len();
					let n_threads = rayon::current_num_threads();
					let work_per_thread = (total_work+n_threads-1) / n_threads;
					let mut work_output: Vec<(F,(R,R))> = Vec::with_capacity(total_work);
					work_output.set_len(total_work);
					work_output.chunks_mut(work_per_thread).enumerate().collect::<Vec<_>>().into_par_iter().for_each(|(i_thread, output)| {
						let start = i_thread * work_per_thread;
						let end = std::cmp::min(start + work_per_thread, total_work);
						(start..end).enumerate().for_each(|(i_output, i_job)| {
							let output_cell = output.get_unchecked_mut(i_output);
							let i = nodes1[i_job / nodes2.len()];
							let j = nodes2[i_job % nodes2.len()];
							if i == j {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_root = union_find.find_immutable(i);
							let j_root = union_find.find_immutable(j);
							if i_root == j_root {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_usize = i.to_usize().unwrap_unchecked();
							let j_usize = j.to_usize().unwrap_unchecked();
							let i_row = data.get_row_view(i_usize);
							let j_row = data.get_row_view(j_usize);
							let d_ij = dist.dist_slice(&i_row, &j_row);
							*output_cell = (d_ij, (i.min(j), i.max(j)));
						});
					});
					work_output
				} else {
					/* Extend the expand queue with neighbor-of-neighbor pairs */
					let nodes1: Vec<R> = graph.iter_neighbors(i).cloned().collect();
					let nodes2: Vec<R> = graph.iter_neighbors(j).cloned().collect();
					let total_work = nodes1.len() + nodes2.len();
					let n_threads = rayon::current_num_threads();
					let work_per_thread = (total_work+n_threads-1) / n_threads;
					let mut work_output: Vec<(F,(R,R))> = Vec::with_capacity(total_work);
					work_output.set_len(total_work);
					work_output.chunks_mut(work_per_thread).enumerate().collect::<Vec<_>>().into_par_iter().for_each(|(i_thread, output)| {
						let start = i_thread * work_per_thread;
						let end = std::cmp::min(start + work_per_thread, total_work);
						(start..end).enumerate().for_each(|(i_output, i_job)| {
							let output_cell = output.get_unchecked_mut(i_output);
							let (i,j) = if i_job < nodes1.len() {
								(nodes1[i_job], j)
							} else {
								(i, nodes2[i_job - nodes1.len()])
							};
							if i == j {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_root = union_find.find_immutable(i);
							let j_root = union_find.find_immutable(j);
							if i_root == j_root {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_usize = i.to_usize().unwrap_unchecked();
							let j_usize = j.to_usize().unwrap_unchecked();
							let i_row = data.get_row_view(i_usize);
							let j_row = data.get_row_view(j_usize);
							let d_ij = dist.dist_slice(&i_row, &j_row);
							*output_cell = (d_ij, (i.min(j), i.max(j)));
						});
					});
					work_output
				};
				/* Insert edges into the queue if not yet observed */
				work_output.into_iter()
				.filter(|(d,_)| *d >= F::zero())
				.for_each(|(d_ij, (i, j))| {
					if observed_edges.observe_edge(i, j) {
						/* Only push if the edge was not already observed */
						expand_queue.push(d_ij, (i.min(j), i.max(j)));
					}
				});
			}
		}
	}
	// println!("{:?}", start_time.elapsed());
	/* Return the dendrogram */
	(dendrogram, core_distances)
}


#[cfg(test)]
mod tests {
	use ndarray::Array2;
	use ndarray_rand::rand_distr::Normal;
	use rand::prelude::Distribution;

	use crate::{cluster::{hnsw_based_dendrogram, hnsw_based_dendrogram_self_joined}, hnsw::HNSWParams};

	#[test]
	fn test_hnsw_based_dendrogram() {
		let (n,d) = (10_000, 3);
		let rng1 = Normal::new(0.0, 1.0).unwrap();
		let rng2 = Normal::new(5.0, 1.0).unwrap();
		let data: Array2<f32> = Array2::from_shape_fn((n, d), |(i,_)| (if i%2 == 0 {rng1} else {rng2}).sample(&mut rand::thread_rng()));
		let start_time = std::time::Instant::now();
		let (dendrogram, core_distances) = hnsw_based_dendrogram::<f32, u32, _, _>(
			&data,
			graphidx::measures::SquaredEuclideanDistance::new(),
			5,
			true,
			false,
			HNSWParams::new(),
		);
		println!("Dendrogram computation took: {:.2?}", start_time.elapsed());
		dendrogram.iter().skip(n-10).for_each(|v| {
			println!("{} {} {} {}", v.0, v.1, v.2, v.3);
		});
		let mean_core_dist = core_distances.iter().sum::<f32>() / core_distances.len() as f32;
		let std_core_dist = core_distances.iter().map(|&x| (x - mean_core_dist).powi(2)).sum::<f32>() / core_distances.len() as f32;
		println!("Mean core distance: {}", mean_core_dist);
		println!("Std core distance: {}", std_core_dist);
	}

	#[test]
	fn test_hnsw_based_dendrogram_self_joined() {
		let (n,d) = (10_000, 3);
		let rng1 = Normal::new(0.0, 1.0).unwrap();
		let rng2 = Normal::new(5.0, 1.0).unwrap();
		let data: Array2<f32> = Array2::from_shape_fn((n, d), |(i,_)| (if i%2 == 0 {rng1} else {rng2}).sample(&mut rand::thread_rng()));
		for query_local in [false, true] {
			let start_time = std::time::Instant::now();
			let (dendrogram, core_distances) = hnsw_based_dendrogram_self_joined::<f32, u32, _, _>(
				&data,
				graphidx::measures::SquaredEuclideanDistance::new(),
				5,
				true,
				false,
				HNSWParams::new(),
				100,
				25,
				query_local,
			);
			println!("Dendrogram computation took: {:.2?}", start_time.elapsed());
			dendrogram.iter().skip(n-10).for_each(|v| {
				println!("{} {} {} {}", v.0, v.1, v.2, v.3);
			});
			let mean_core_dist = core_distances.iter().sum::<f32>() / core_distances.len() as f32;
			let std_core_dist = core_distances.iter().map(|&x| (x - mean_core_dist).powi(2)).sum::<f32>() / core_distances.len() as f32;
			println!("Mean core distance: {}", mean_core_dist);
			println!("Std core distance: {}", std_core_dist);
		}
	}
}

