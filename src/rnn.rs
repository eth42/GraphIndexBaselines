use graphidx::{
	data::MatrixDataSource, graphs::{DirLoLGraph, Graph, WeightedGraph}, indices::{GreedyCappedSingleGraphIndex, GreedySingleGraphIndex}, measures::Distance, param_struct, random::RandomPermutationGenerator, types::{Float, SyncFloat, SyncUnsignedInteger}
};
use rayon::{current_num_threads, iter::{ParallelBridge, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use std::{mem::swap, sync::{atomic::{AtomicUsize,Ordering::Relaxed}, Mutex}};
use foldhash::HashSet;

use crate::util::{random_unique_usizes_except, random_usize_pairs, remove_duplicates_with_key};


pub struct RNNBuildGraph<R: SyncUnsignedInteger, F: SyncFloat> {
	adjacency: Vec<Vec<(F,R,bool)>>,
	n_edges: usize
}
impl<R: SyncUnsignedInteger, F: SyncFloat> RNNBuildGraph<R,F> {
	#[inline(always)]
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
	#[allow(unused)]
	fn set_flag(&mut self, vertex1: R, vertex2: R, flag: bool) {
		let neighbor_idx = self.find_edge(vertex1, vertex2);
		if neighbor_idx.is_some() {
			let v1usize = unsafe{vertex1.to_usize().unwrap_unchecked()};
			self.adjacency[v1usize][unsafe{neighbor_idx.unwrap_unchecked()}].2 = flag;
		}
	}
	#[inline(always)]
	fn set_flag_all_neighbors(&mut self, vertex: R, flag: bool) {
		let vertex = unsafe{vertex.to_usize().unwrap_unchecked()};
		self.adjacency[vertex].iter_mut().for_each(|(_,_,f)| *f = flag);
	}
	#[inline(always)]
	fn get_adj(&self, vertex: R) -> &Vec<(F,R,bool)> {
		let vertex = unsafe{vertex.to_usize().unwrap_unchecked()};
		&self.adjacency[vertex]
	}
	#[inline(always)]
	fn get_adj_mut(&mut self, vertex: R) -> &mut Vec<(F,R,bool)> {
		let vertex = unsafe{vertex.to_usize().unwrap_unchecked()};
		&mut self.adjacency[vertex]
	}
	#[inline(always)]
	fn clear_adj(&mut self, vertex: R) {
		let vertex = unsafe{vertex.to_usize().unwrap_unchecked()};
		self.adjacency[vertex].clear();
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat> Graph<R> for RNNBuildGraph<R,F> {
	#[inline(always)]
	fn reserve(&mut self, n_vertices: usize) {
		self.adjacency.reserve(n_vertices);
	}
	#[inline(always)]
	fn n_vertices(&self) -> usize {
		self.adjacency.len()
	}
	#[inline(always)]
	fn n_edges(&self) -> usize {
		self.n_edges
	}
	#[inline(always)]
	fn neighbors(&self, vertex: R) -> Vec<R> {
		let vertex = unsafe{vertex.to_usize().unwrap_unchecked()};
		self.adjacency[vertex].iter().map(|&(_,v,_)| v).collect()
	}
	#[inline(always)]
	fn foreach_neighbor<Fun: FnMut(&R)>(&self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(|v|f(&v.1));
	}
	#[inline(always)]
	fn foreach_neighbor_mut<Fun: FnMut(&mut R)>(&mut self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(|v|f(&mut v.1));
	}
	#[inline(always)]
	fn iter_neighbors<'a>(&'a self, vertex: R) -> impl Iterator<Item=&'a R> {
		unsafe {
			self.adjacency.get_unchecked(vertex.to_usize().unwrap_unchecked()).iter().map(|v|&v.1)
		}
	}
	#[inline(always)]
	fn add_node(&mut self) {
		self.adjacency.push(Vec::new());
	}
	#[inline(always)]
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(Vec::with_capacity(capacity));
	}
	#[inline(always)]
	fn add_edge(&mut self, _vertex1: R, _vertex2: R) {
		panic!("Cannot add edge without weight to a weighted graph");
	}
	#[inline(always)]
	fn remove_edge_by_index(&mut self, vertex: R, index: usize) {
		let vertex = unsafe{vertex.to_usize().unwrap_unchecked()};
		self.adjacency[vertex].swap_remove(index);
		self.n_edges -= 1;
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat> WeightedGraph<R,F> for RNNBuildGraph<R,F> {
	#[inline(always)]
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
		let vertex1 = unsafe{vertex1.to_usize().unwrap_unchecked()};
		self.adjacency[vertex1].iter().find(|&&(_,v,_)| v == vertex2).unwrap().0
	}
	#[inline(always)]
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F) {
		let vertex1 = unsafe{vertex1.to_usize().unwrap_unchecked()};
		self.adjacency[vertex1].push((weight,vertex2,true));
		debug_assert!(self.adjacency[vertex1][self.adjacency[vertex1].len()-1] == (weight, vertex2, true));
		self.n_edges += 1;
	}
	#[inline(always)]
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
		let mut neighbors = Vec::new();
		let mut weights = Vec::new();
		for &(w,v,_) in &self.adjacency[vertex.to_usize().unwrap()] {
			neighbors.push(v);
			weights.push(w);
		}
		(weights, neighbors)
	}
	#[inline(always)]
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)> {
		self.adjacency[vertex.to_usize().unwrap()].iter().map(|&(v,w,_)| (v,w)).collect()
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight<Fun: FnMut(&F, &R)>(&self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(|v| f(&v.0,&v.1));
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight_mut<Fun: FnMut(&mut F, &mut R)>(&mut self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(|v| f(&mut v.0,&mut v.1));
	}
}

#[inline(always)]
fn make_random_graph<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	Dist: Distance<F>+Sync+Send,
	M: MatrixDataSource<F>+Sync,
>(mat: &M, graph: &mut RNNBuildGraph<R,F>, dist: &Dist, out_degree: usize) {
	let n_data = mat.n_rows();
	graph.reserve(n_data);
	(0..n_data).for_each(|_| graph.add_node_with_capacity(out_degree));
	/* Partition the workload into equal size for each thread */
	let n_threads = current_num_threads();
	let thread_chunk_size = (n_data + n_threads - 1) / n_threads;
	unsafe {
		(0..n_data).step_by(thread_chunk_size)
		.map(|start| (start, (start+thread_chunk_size).min(n_data)))
		.par_bridge()
		.for_each(|(start, end)| {
			let unsafe_graph_ref = std::ptr::addr_of!(*graph) as *mut RNNBuildGraph<R,F>;
			(start..end)
			.for_each(|iusize| {
				let i = R::from_usize(iusize).unwrap_unchecked();
				let neighbors = random_unique_usizes_except(n_data, out_degree, iusize);
				(*unsafe_graph_ref).get_adj_mut(i).extend(neighbors.into_iter().map(|jusize| {
					let j = R::from_usize(jusize).unwrap_unchecked();
					let ij_dist = if M::SUPPORTS_ROW_VIEW {
						dist.dist_slice(&mat.get_row_view(iusize), &mat.get_row_view(jusize))
					} else {
						dist.dist(&mat.get_row(iusize), &mat.get_row(jusize))
					};
					(ij_dist, j, true)
				}));
			});
		});
		graph.n_edges += out_degree*n_data;
	}
}
#[allow(unused)]
#[inline(always)]
fn add_random_edges<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	Dist: Distance<F>+Sync+Send,
	M: MatrixDataSource<F>+Sync,
>(mat: &M, graph: &mut RNNBuildGraph<R,F>, dist: &Dist, n_edges: usize) {
	let n_nodes = graph.n_vertices();
	let n_threads = current_num_threads();
	let chunk_size = (n_edges + n_threads - 1) / n_threads;
	let random_pairs = random_usize_pairs(n_nodes, n_edges);
	let mut random_edges = random_pairs.par_chunks(chunk_size).map(|chunk| {
		chunk.iter().filter(|(u,v)| u!=v).map(|&(u,v)| {
			let dist = if M::SUPPORTS_ROW_VIEW {
				dist.dist_slice(&mat.get_row_view(u), &mat.get_row_view(v))
			} else {
				dist.dist(&mat.get_row(u), &mat.get_row(v))
			};
			unsafe{(R::from_usize(u).unwrap_unchecked(),R::from_usize(v).unwrap_unchecked(),dist)}
		}).collect::<Vec<_>>()
	}).flatten().collect::<Vec<_>>();
	batch_add_edges(graph, false, &mut random_edges);
}
#[inline(always)]
fn batch_set_flags<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	G: WeightedGraph<R,F>+Sync,
>(graph: &mut G, set_flag_ids: Vec<usize>) {
	let thread_batch_size = (set_flag_ids.len()+current_num_threads()-1)/current_num_threads();
	set_flag_ids.chunks(thread_batch_size)
	.par_bridge().for_each(|chunk| {
		chunk.into_iter().for_each(|&u| {
			unsafe{
				let unsafe_graph_ref = std::ptr::addr_of!(*graph) as *mut RNNBuildGraph<R,F>;
				(*unsafe_graph_ref).set_flag_all_neighbors(R::from_usize(u).unwrap_unchecked(), false);
			}
		});
	});
}
#[allow(unused)]
#[inline(always)]
fn batch_remove_edges<
	R: SyncUnsignedInteger,
	F: SyncFloat,
>(graph: &mut RNNBuildGraph<R,F>, presorted: bool, remove_edges: &mut Vec<(R,R)>) {
	if !presorted { remove_edges.par_sort_unstable_by(|(u1,_),(u2,_)| u1.cmp(u2)); }
	let removed_edge_count = AtomicUsize::new(0);
	remove_edges.par_chunk_by(|(u1,_),(u2,_)| u1==u2).for_each(|chunk| {
		debug_assert!(chunk.len() > 0);
		unsafe{
			let unsafe_graph_ref = std::ptr::addr_of!(*graph) as *mut RNNBuildGraph<R,F>;
			let u = chunk[0].0;
			let adj = (*unsafe_graph_ref).get_adj_mut(u);
			chunk.into_iter().for_each(|&(_,v)| {
				let pos = adj.iter().enumerate().filter(|&(_,&(_,v2,_))| v==v2).next().map(|(pos,_)| pos);
				if pos.is_some() {
					adj.swap_remove(pos.unwrap());
					removed_edge_count.fetch_add(1, Relaxed);
				}
			});
		}
	});
	graph.n_edges -= removed_edge_count.load(Relaxed);
}
#[inline(always)]
fn batch_add_edges<
	R: SyncUnsignedInteger,
	F: SyncFloat,
>(graph: &mut RNNBuildGraph<R,F>, presorted: bool, add_edges: &mut Vec<(R,R,F)>) {
	if !presorted { add_edges.par_sort_unstable_by(|(u1,_,_),(u2,_,_)| u1.cmp(u2)); }
	let added_edge_count = AtomicUsize::new(0);
	add_edges.par_chunk_by(|(u1,_,_),(u2,_,_)| u1==u2).for_each(|chunk| {
		debug_assert!(chunk.len() > 0);
		unsafe{
			let unsafe_graph_ref = std::ptr::addr_of!(*graph) as *mut RNNBuildGraph<R,F>;
			let u = chunk[0].0;
			let adj = (*unsafe_graph_ref).get_adj_mut(u);
			adj.reserve(chunk.len());
			chunk.into_iter().for_each(|&(_,v,dist)| {
				if adj.iter().find(|&&(_,v2,_)| v==v2).is_none() {
					adj.push((dist,v,true));
					added_edge_count.fetch_add(1, Relaxed);
				}
			});
		}
	});
	graph.n_edges += added_edge_count.load(Relaxed);
}
#[inline(always)]
fn make_greedy_index<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	M: MatrixDataSource<F>,
	Dist: Distance<F>,
>(graph: RNNBuildGraph<R,F>, mat: M, dist: Dist) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
	GreedySingleGraphIndex::new(mat, graph.as_dir_lol_graph(), dist, None)
}
#[inline(always)]
fn make_greedy_capped_index<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	M: MatrixDataSource<F>,
	Dist: Distance<F>,
>(graph: RNNBuildGraph<R,F>, mat: M, dist: Dist, max_frontier_size: usize) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
	GreedyCappedSingleGraphIndex::new(mat, graph.as_dir_lol_graph(), dist, max_frontier_size, None)
}
#[inline(always)]
fn tri_to_cos<F: Float>(uv: F, uw: F, vw: F, dist_is_sq: bool) -> F {
	let (uv2, uw2, vw2, uvuw) = if !dist_is_sq {
		(uv*uv, uw*uw, vw*vw, uv*uw)
	} else {
		(uv, uw, vw, <F as num::Float>::sqrt(uv*uw).max(F::zero()))
	};
	let denominator = uvuw+uvuw;
	if denominator > F::zero() { (uv2+uw2-vw2)/denominator } else { -F::one() }
}




macro_rules! with_guard {
	($self:ident, $num:ident, $body:expr) => {{
		let guard = $self.node_locks[$num].lock().unwrap();
		let ret = $body;
		drop(guard);
		ret
	}};
}


pub trait RNNStyleBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> where Self: Sized {
	type Params;
	fn _mut_graph(&mut self) -> &mut RNNBuildGraph<R,F>;
	fn _dist(&self) -> &Dist;
	fn _initial_degree(&self) -> usize;
	fn _into_graph_dist(self) -> (RNNBuildGraph<R,F>, Dist);
	#[inline(always)]
	fn _get_dist<M: MatrixDataSource<F>>(&self, mat: &M, i: usize, j: usize) -> F {
		if M::SUPPORTS_ROW_VIEW {
			self._dist().dist_slice(&mat.get_row_view(i), &mat.get_row_view(j))
		} else {
			self._dist().dist(&mat.get_row(i), &mat.get_row(j))
		}
	}
	/// Creates a new RNN style builder instance, allocates some memory and stores the parameters
	fn new<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self;
	/// Randomly initializes the graph
	#[inline(always)]
	fn init_random<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let (dist, initial_degree) = (self._dist() as *const Dist, self._initial_degree());
		make_random_graph(mat, self._mut_graph(), unsafe{dist.as_ref().unwrap()}, initial_degree);
	}
	/// Trains the graph by applying RNN descent or similar routine
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M);
	/// Creates a new RNN style builder instance using a given graph
	#[inline(always)]
	fn from_graph<M: MatrixDataSource<F>+Sync, G: Graph<R>>(mat: &M, graph: &G, dist: Dist, params: Self::Params) -> Self {
		let mut builder = Self::new(&mat, dist.clone(), params);
		let n_vertices = graph.n_vertices();
		let rnn_graph = builder._mut_graph();
		rnn_graph.reserve(n_vertices);
		unsafe {
			(0..n_vertices)
			.map(|u| (u,R::from_usize(u).unwrap_unchecked()))
			.for_each(|(u_usize, u)| {
				let u_degree = graph.degree(u);
				rnn_graph.add_node_with_capacity(u_degree);
				graph.foreach_neighbor(u, |&v| {
					let v_usize = v.to_usize().unwrap_unchecked();
					let dist = if M::SUPPORTS_ROW_VIEW {
						dist.dist_slice(mat.get_row_view(u_usize), mat.get_row_view(v_usize))
					}else {
						dist.dist(&mat.get_row(u_usize), &mat.get_row(v_usize))
					};
					rnn_graph.get_adj_mut(u).push((dist,v,true));
				});
			});
		}
		builder.train(mat);
		builder
	}
	/// Creates a new RNN style builder instance using a given weighted graph
	#[inline(always)]
	fn from_weighted_graph<M: MatrixDataSource<F>+Sync, G: WeightedGraph<R,F>>(mat: &M, graph: &G, dist: Dist, params: Self::Params) -> Self {
		let mut builder = Self::new(&mat, dist, params);
		let n_vertices = graph.n_vertices();
		let rnn_graph = builder._mut_graph();
		rnn_graph.reserve(n_vertices);
		unsafe {
			(0..n_vertices)
			.map(|u| R::from_usize(u).unwrap_unchecked())
			.for_each(|u| {
				let u_degree = graph.degree(u);
				rnn_graph.add_node_with_capacity(u_degree);
				graph.foreach_neighbor_with_zipped_weight(u, |&dist, &v| {
					rnn_graph.get_adj_mut(u).push((dist,v,true));
				});
			});
		}
		builder.train(mat);
		builder
	}
	/// Transforms a trained RNN style builder into a greedy index
	#[inline(always)]
	fn into_greedy<M: MatrixDataSource<F>+Sync>(self, mat: M) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
		let (graph, dist) = self._into_graph_dist();
		make_greedy_index(graph, mat, dist)
	}
	/// Transforms a trained RNN style builder into a capped greedy index
	#[inline(always)]
	fn into_greedy_capped<M: MatrixDataSource<F>+Sync>(self, mat: M, max_frontier_size: usize) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
		let (graph, dist) = self._into_graph_dist();
		make_greedy_capped_index(graph, mat, dist, max_frontier_size)
	}
	/// Shorthand for new, init_random, train, into_greedy
	#[inline(always)]
	fn build<M: MatrixDataSource<F>+Sync>(mat: M, dist: Dist, params: Self::Params) -> GreedySingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
		let mut builder = Self::new(&mat, dist, params);
		builder.init_random(&mat);
		builder.train(&mat);
		builder.into_greedy(mat)
	}
	/// Shorthand for new, init_random, train, into_greedy_capped
	#[inline(always)]
	fn build_capped<M: MatrixDataSource<F>+Sync>(mat: M, dist: Dist, params: Self::Params, max_frontier_size: usize) -> GreedyCappedSingleGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
		let mut builder = Self::new(&mat, dist, params);
		builder.init_random(&mat);
		builder.train(&mat);
		builder.into_greedy_capped(mat, max_frontier_size)
	}
}


param_struct!(RNNParams[Copy, Clone] {
	initial_degree: usize = 100,
	reduce_degree: usize = 50,
	n_outer_loops: usize = 4,
	n_inner_loops: usize = 15,
	concurrent_batch_size: usize = 100,
});
/* Locking-free implementation computing reverse edges on-demand */
pub struct RNNDescentBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	params: RNNParams,
	node_locks: Vec<Mutex<()>>,
	rev_edge_cache: Vec<Vec<(R,F,bool)>>,
	add_edge_cache: Vec<(R,R,F)>,
	graph: RNNBuildGraph<R,F>,
	dist: Dist,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> RNNDescentBuilder<R, F, Dist> {
	fn update_neighbors<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let thread_chunk_size = self.params.concurrent_batch_size;
		let global_chunk_size = thread_chunk_size*current_num_threads();
		let n_data = self.n_data;
		let perm_gen = RandomPermutationGenerator::new(n_data, 4);
		let removed_edge_count = AtomicUsize::new(0);
		(0..n_data).step_by(global_chunk_size)
		.map(|start| (start, (start+global_chunk_size).min(n_data)))
		.for_each(|(start, end)| {
			/* Compute edges to remove and to add in parallel */
			/* Working in chunks to remove the number of spawned threads and therefore overhead */
			let add_edges_list = (start..end).step_by(thread_chunk_size)
			.map(|lstart| (lstart, (lstart+thread_chunk_size).min(end)))
			.par_bridge()
			.map(|(lstart, lend)| {
				let unsafe_graph_ref = std::ptr::addr_of!(self.graph) as *mut RNNBuildGraph<R,F>;

				let mut new_neighbors: Vec<(F,usize,R,bool)> = vec![];
				let mut add_edges: Vec<(R,R,F)> = vec![];
				let mut remove_ids: Vec<usize> = vec![];

				(lstart..lend).map(|u| perm_gen.apply_rounds(u))
				.map(|u| (u,unsafe{R::from_usize(u).unwrap_unchecked()}))
				.for_each(|(_, u)| {
					let old_neighbors = unsafe{(*unsafe_graph_ref).get_adj_mut(u)};
					old_neighbors.sort_unstable_by(|(d1,_,_),(d2,_,_)| unsafe{d1.partial_cmp(d2).unwrap_unchecked()});
					new_neighbors.clear();
					remove_ids.clear();
					old_neighbors.into_iter().enumerate().for_each(|(i,&mut (vdist, v, vflag))| {
						let vusize = unsafe{R::to_usize(&v).unwrap_unchecked()};
						let mut keep_neighbor = true;
						for &(_, wusize, w, wflag) in new_neighbors.iter() {
							/* If both neighbors are "old" continue */
							if !vflag && !wflag { continue; }
							
							let vwdist = self._get_dist(mat, vusize,wusize);
							if vwdist <= vdist {
								// prune by RNG rule
								keep_neighbor = false;
								// insert connection w -> v instead:
								add_edges.push((w, v, vwdist));
								break;
							}
						}
						if keep_neighbor {
							new_neighbors.push((vdist,vusize,v,vflag));
						} else {
							remove_ids.push(i);
						}
					});
					remove_ids.iter().rev().for_each(|&i| _=old_neighbors.swap_remove(i));
					removed_edge_count.fetch_add(remove_ids.len(), Relaxed);
				});

				add_edges
			}).collect::<Vec<_>>();
			/* Collect all partial remove and add edges into one big vec, presort and apply */
			let n_add_edges = add_edges_list.iter().map(|ae| ae.len()).sum();
			let add_edges = &mut self.add_edge_cache;
			add_edges.clear();
			add_edges.reserve(n_add_edges);
			add_edges_list.into_iter().for_each(|ae| add_edges.extend(ae));
			/* Presort and remove duplicate insertions (duplicates in removals must stay) */
			unsafe {
				add_edges.par_sort_unstable_by(|(u1,v1, _),(u2,v2, _)| {
					let u_cmp = u1.partial_cmp(u2).unwrap_unchecked();
					if u_cmp.is_eq() { v1.partial_cmp(v2).unwrap_unchecked() } else { u_cmp }
				});
			}
			/* This is not important enough to justify single-thread work, we test for duplicate edges later on anyways */
			// remove_duplicates(add_edges);
			/* Apply edge changes in parallel and lock free */
			batch_set_flags(&mut self.graph, (start..end).map(|i|perm_gen.apply_rounds(i)).collect());
			batch_add_edges(&mut self.graph, true, add_edges);
		});
		/* Reduce the edge count to keep consistency */
		self.graph.n_edges -= removed_edge_count.load(Relaxed);
	}
	fn add_reverse_edges(&mut self) {
		/* Partition the workload into equal size for each thread */
		let thread_chunk_size = (self.n_data + current_num_threads() - 1) / current_num_threads();
		let n_data = self.n_data;
		let reduce_degree = self.params.reduce_degree;
		unsafe {
			/* Clear reverse edge cache */
			self.rev_edge_cache.par_chunks_mut(thread_chunk_size)
			.for_each(|chunk| {
				chunk.iter_mut().for_each(|rev_cache| rev_cache.clear());
			});
			/* Build reverse edge cache */
			(0..n_data).step_by(thread_chunk_size).par_bridge()
			.map(|start| (start, (start+thread_chunk_size).min(n_data)))
			.for_each(|(start, end)| {
				let mut neighbor_cache: Vec<(F,R,bool)> = Vec::with_capacity(2*reduce_degree);
				let unsafe_cache_ref = &mut neighbor_cache as *mut Vec<(F,R,bool)>;
				let unsafe_rev_cache_ref = std::ptr::addr_of!(self.rev_edge_cache) as *mut Vec<Vec<(F,R,bool)>>;
				(start..end).for_each(|uusize| {
					let u = R::from_usize(uusize).unwrap_unchecked();
					let old_neighbors = unsafe_cache_ref.as_mut().unwrap_unchecked();
					old_neighbors.clear();
					with_guard!(self, uusize, {
						old_neighbors.extend(self.graph.get_adj(u));
						(&mut *unsafe_rev_cache_ref).get_unchecked_mut(uusize).extend(self.graph.get_adj(u));
					});
					old_neighbors.iter().for_each(|&(dist, v, _)| {
						let vusize = R::to_usize(&v).unwrap_unchecked();
						with_guard!(self, vusize, (&mut *unsafe_rev_cache_ref).get_unchecked_mut(vusize).push((dist,u,true)));
					});
				})
			});
			/* Check and keep reverse edges */
			let n_total_edges = (0..n_data).step_by(thread_chunk_size).par_bridge()
			.map(|start| (start, (start+thread_chunk_size).min(n_data)))
			.map(|(start, end)| {
				let unsafe_graph_ref = std::ptr::addr_of!(self.graph) as *mut RNNBuildGraph<R,F>;
				let unsafe_rev_cache_ref = std::ptr::addr_of!(self.rev_edge_cache) as *mut Vec<Vec<(F,R,bool)>>;
				(start..end).map(|uusize| {
					let u = R::from_usize(uusize).unwrap_unchecked();
					let joined_neighbors = (&mut *unsafe_rev_cache_ref).get_unchecked_mut(uusize);
					/* Sorting such that: */
					/* - Closer neighbors are first */
					/* - Smaller IDs of equally far neighbors are first */
					/* - "not new" flag comes first if everything else is equal */
					joined_neighbors.sort_unstable_by(|(d1,v1,f1),(d2,v2,f2)| {
						let dist_cmp = d1.partial_cmp(d2).unwrap_unchecked();
						if dist_cmp.is_eq() {
							let id_cmp = v1.partial_cmp(v2).unwrap_unchecked();
							if id_cmp.is_eq() { f1.cmp(f2) } else { id_cmp }
						} else { dist_cmp }
					});
					remove_duplicates_with_key(joined_neighbors, |(_,v,_)| v);
					(*unsafe_graph_ref).clear_adj(u);
					let n_edges = reduce_degree.min(joined_neighbors.len());
					(*unsafe_graph_ref).get_adj_mut(u).extend(joined_neighbors[0..n_edges].iter());
					n_edges
				}).sum::<usize>()
			}).sum();
			self.graph.n_edges = n_total_edges;
		}
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> RNNStyleBuilder<R, F, Dist> for RNNDescentBuilder<R, F, Dist> {
	type Params = RNNParams;
	#[inline(always)]
	fn _mut_graph(&mut self) -> &mut RNNBuildGraph<R,F> { &mut self.graph }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _initial_degree(&self) -> usize { self.params.initial_degree }
	#[inline(always)]
	fn _into_graph_dist(self) -> (RNNBuildGraph<R,F>, Dist) { (self.graph, self.dist) }
	fn new<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let graph = RNNBuildGraph::new();
		let reduce_degree = params.reduce_degree;
		Self {
			_phantom: std::marker::PhantomData,
			n_data,
			params,
			node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
			rev_edge_cache: (0..n_data).map(|_| Vec::with_capacity(2*reduce_degree)).collect(),
			add_edge_cache: Vec::new(),
			graph,
			dist,
		}
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		self.graph.reserve(self.n_data);
		let (n_outer_loops, n_inner_loops) = (self.params.n_outer_loops, self.params.n_outer_loops);
		(0..n_outer_loops).for_each(|i| {
			(0..n_inner_loops).for_each(|_j| {
				self.update_neighbors(mat);
				#[cfg(debug_assertions)]
				println!("Outer loop: {:?}, Inner loop: {:?}, Nodes: {:?}, Edges: {:?}", i, _j, self.graph.n_vertices(), self.graph.n_edges());
			});
			if i != n_outer_loops-1 {
				self.add_reverse_edges();
				#[cfg(debug_assertions)]
				println!("Outer loop: {:?}, Inner loop: {:?}, Nodes: {:?}, Edges: {:?}", i, n_inner_loops, self.graph.n_vertices(), self.graph.n_edges());
			}
		});
		/* Sanity check to ensure the graph is simple directed */
		// self.remove_duplicate_edges();
	}
}


param_struct!(RNNEgoParams[Copy, Clone] {
	initial_degree: usize = 100,
	n_loops: usize = 4,
	concurrent_batch_size: usize = 100,
	radius: usize = 2,
});
impl RNNEgoParams {
	pub fn extend_rnn(params: RNNParams) -> Self {
		Self{
			initial_degree: params.initial_degree,
			n_loops: params.n_outer_loops * params.n_inner_loops,
			concurrent_batch_size: params.concurrent_batch_size,
			radius: 2,
		}
	}
}
pub struct RNNEgoDescentBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	params: RNNEgoParams,
	node_locks: Vec<Mutex<()>>,
	graph: RNNBuildGraph<R,F>,
	dist: Dist,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> RNNEgoDescentBuilder<R, F, Dist> {
	fn update_neighbors<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let n_threads = current_num_threads();
		let thread_chunk_size = self.params.concurrent_batch_size;
		let global_chunk_size = thread_chunk_size*n_threads;
		let (n_data, radius) = (self.n_data, self.params.radius);
		let perm_gen = RandomPermutationGenerator::new(n_data, 4);
		(0..n_data).step_by(global_chunk_size)
		.map(|start| (start, (start+global_chunk_size).min(n_data)))
		.for_each(|(start, end)| {
			let add_edge_count = AtomicUsize::new(0);
			let remove_edge_count = AtomicUsize::new(0);
			/* Compute edges to remove and to add in parallel */
			/* Working in chunks to remove the number of spawned threads and therefore overhead */
			(start..end).step_by(thread_chunk_size)
			.map(|lstart| (lstart, (lstart+thread_chunk_size).min(end)))
			.par_bridge()
			.for_each(|(lstart, lend)| {
				let unsafe_graph_ref = std::ptr::addr_of!(self.graph) as *mut RNNBuildGraph<R,F>;

				let mut kept_neighbor_cache: Vec<(F,R,bool)> = vec![];
				let mut visited: HashSet<R> = HashSet::default();
				let mut work_queue: Vec<R> = Vec::new();
				let mut next_work_queue: Vec<R> = Vec::new();
				let mut found_ids = Vec::new();

				(lstart..lend).map(|u| perm_gen.apply_rounds(u))
				.map(|u| (u,unsafe{R::from_usize(u).unwrap_unchecked()}))
				.for_each(|(uusize, u)| {
					visited.clear();
					work_queue.clear();
					found_ids.clear();
					with_guard!(self, uusize, {
						let graph_ref = unsafe{&mut *unsafe_graph_ref};
						visited.insert(u);
						work_queue.push(u);
						for _ in 0..radius {
							next_work_queue.clear();
							work_queue.iter().for_each(|&v| {
								graph_ref.get_adj(v).iter().for_each(|&(_,n,_)| {
									if !visited.contains(&n) {
										visited.insert(n);
										next_work_queue.push(n);
										found_ids.push(n);
									}
								});
							});
							swap(&mut work_queue, &mut next_work_queue);
						}
					});
					let mut u_neighbor_cands: Vec<(usize, R, F)> = found_ids.iter()
					.map(|&v| {
						let vusize = unsafe{R::to_usize(&v).unwrap_unchecked()};
						let uvdist = self._get_dist(mat, uusize,vusize);
						(vusize, v, uvdist)
					}).collect();
					u_neighbor_cands.sort_unstable_by(|(_,_,d1),(_,_,d2)| d1.partial_cmp(d2).unwrap());
					kept_neighbor_cache.clear();
					u_neighbor_cands.iter().for_each(|&(vusize, v, vdist)| {
						if kept_neighbor_cache.len() >= 50 { return; }
						let mut keep_neighbor = true;
						for &(_wdist, w, _) in &kept_neighbor_cache {
							let wusize = unsafe{R::to_usize(&w).unwrap_unchecked()};
							let vwdist = self._get_dist(mat, vusize,wusize);
							if vwdist < vdist {
								// prune by RNG rule
								keep_neighbor = false;
								break;
							}
						}
						if keep_neighbor {
							kept_neighbor_cache.push((vdist, v, true));
						}
					});
					with_guard!(self, uusize, {
						let adj = unsafe{(*unsafe_graph_ref).get_adj_mut(u)};
						remove_edge_count.fetch_add(adj.len(), Relaxed);
						swap(adj, &mut kept_neighbor_cache);
						add_edge_count.fetch_add(adj.len(), Relaxed);
					});
				});
			});
			/* Update the edge count to keep consistency */
			// println!("Remove edges: {:?}, Add edges: {:?}", remove_edge_count.load(Relaxed), add_edge_count.load(Relaxed));
			self.graph.n_edges -= remove_edge_count.load(Relaxed);
			self.graph.n_edges += add_edge_count.load(Relaxed);
		});
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> RNNStyleBuilder<R, F, Dist> for RNNEgoDescentBuilder<R, F, Dist> {
	type Params = RNNEgoParams;
	#[inline(always)]
	fn _mut_graph(&mut self) -> &mut RNNBuildGraph<R,F> { &mut self.graph }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _initial_degree(&self) -> usize { self.params.initial_degree }
	#[inline(always)]
	fn _into_graph_dist(self) -> (RNNBuildGraph<R,F>, Dist) { (self.graph, self.dist) }
	fn new<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let graph = RNNBuildGraph::new();
		Self {
			_phantom: std::marker::PhantomData,
			n_data,
			params,
			node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
			graph,
			dist,
		}
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		assert_eq!(self.graph.n_edges(), (0..self.graph.n_vertices()).map(|u| self.graph.get_adj(R::from_usize(u).unwrap()).len()).sum::<usize>());
		println!("After initialization, Nodes: {:?}, Edges: {:?}", self.graph.n_vertices(), self.graph.n_edges());
		(0..self.params.n_loops).for_each(|_i| {
			if _i > 0 {
				let mut add_edges = (0..self.graph.n_vertices()).map(|u| {
					let u = R::from_usize(u).unwrap();
					self.graph.get_adj(u).iter()
					.map(move |&(dist,v,_)| (v,u,dist))
				}).flatten().collect();
				batch_add_edges(&mut self.graph, false, &mut add_edges);
			}
			self.update_neighbors(mat);
			// #[cfg(debug_assertions)]
			println!("Loop: {:?}, Nodes: {:?}, Edges: {:?}", _i, self.graph.n_vertices(), self.graph.n_edges());
		});
		/* Sanity check to ensure the graph is simple directed */
		// self.remove_duplicate_edges();
	}
}



pub fn bruteforce_rng_edges<F: SyncFloat, M: MatrixDataSource<F>+Sync>(data: &M) -> Vec<(usize, usize)> {
	use ndarray::Array2;
	use graphidx::measures::SquaredEuclideanDistance;
	let mut result = Vec::new();
	let n_data = data.n_rows();
	let n_threads = current_num_threads();
	let chunk_size = (n_data+n_threads-1)/n_threads;
	let dist = SquaredEuclideanDistance::new();
	/* Bruteforce all pairwise distances of the dataset */
	let dist_mat = Array2::from_elem((n_data,n_data), F::zero());
	let perm_gen = RandomPermutationGenerator::new(n_data, 4);
	(0..n_data).step_by(chunk_size)
	.map(|start| (start, (start+chunk_size).min(n_data)))
	.par_bridge()
	.for_each(|(start, end)| {
		let unsafe_dist_mat_ref = std::ptr::addr_of!(dist_mat) as *mut Array2<F>;
		(start..end).for_each(|i| {
			let i = perm_gen.apply_rounds(i);
			let p = data.get_row_view(i);
			(0..i).for_each(|j: usize| {
				let q = data.get_row_view(j);
				let dist = dist.dist_slice(&p, &q);
				unsafe {
					(&mut *unsafe_dist_mat_ref)[[i,j]] = dist;
					(&mut *unsafe_dist_mat_ref)[[j,i]] = dist;
				}
			});
		});
	});
	let mut dist_cache = vec![(0, F::zero()); n_data];
	let mut edge_cache: Vec<usize> = Vec::new();
	(0..n_data).for_each(|i| {
		/* Bruteforce compute all distances */
		(0..n_data).step_by(chunk_size)
		.map(|start| (start, (start+chunk_size).min(n_data)))
		.zip(dist_cache.chunks_mut(chunk_size))
		.par_bridge()
		.for_each(|((start, end), dist_chunk)| {
			(start..end).for_each(|j| dist_chunk[j-start] = (j, dist_mat[[i,j]]));
		});
		dist_cache.par_sort_unstable_by(|(_,d1),(_,d2)| d1.partial_cmp(d2).unwrap());
		/* Find edges by RNG rule */
		edge_cache.clear();
		dist_cache.iter().skip(1).for_each(|&(j,_)| {
			if edge_cache.iter().all(|&k| dist_mat[[j,k]] > dist_mat[[i,j]]) {
				edge_cache.push(j);
			}
		});
		result.extend(edge_cache.iter().map(|&j| (i,j)));
	});
	result
}


param_struct!(SENParams[Copy, Clone]<F: SyncFloat> {
	initial_degree: usize = 100,
	reduce_degree: usize = 50,
	n_outer_loops: usize = 4,
	n_inner_loops: usize = 15,
	concurrent_batch_size: usize = 100,
	max_cos: F = F::from(0.5).unwrap(),
	dist_is_sq: bool = true,
	prune_non_sen_edges: bool = true,
	verify_sen_edges: bool = false,
});
impl<F: SyncFloat> SENParams<F> {
	pub fn extend_rnn(params: RNNParams) -> Self {
		Self{
			initial_degree: params.initial_degree,
			reduce_degree: params.reduce_degree,
			n_outer_loops: params.n_outer_loops,
			n_inner_loops: params.n_inner_loops,
			concurrent_batch_size: params.concurrent_batch_size,
			max_cos: F::from(0.5).unwrap(),
			dist_is_sq: true,
			prune_non_sen_edges: true,
			verify_sen_edges: false,
		}
	}
}

/* Locking-free implementation computing reverse edges on-demand */
pub struct SENDescentBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	params: SENParams<F>,
	node_locks: Vec<Mutex<()>>,
	rev_edge_cache: Vec<Vec<(F,R,bool)>>,
	add_edge_cache: Vec<(R,R,F)>,
	graph: RNNBuildGraph<R,F>,
	dist: Dist,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> SENDescentBuilder<R, F, Dist> {
	fn update_neighbors<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let thread_chunk_size = self.params.concurrent_batch_size;
		let global_chunk_size = thread_chunk_size*current_num_threads();
		let (n_data, max_cos, dist_is_sq, verify_sen_edges) = (self.n_data, self.params.max_cos, self.params.dist_is_sq, self.params.verify_sen_edges);
		let perm_gen = RandomPermutationGenerator::new(n_data, 4);
		let removed_edge_count = AtomicUsize::new(0);
		(0..n_data).step_by(global_chunk_size)
		.map(|start| (start, (start+global_chunk_size).min(n_data)))
		.for_each(|(start, end)| {
			/* Compute edges to remove and to add in parallel */
			/* Working in chunks to remove the number of spawned threads and therefore overhead */
			let add_edges_list = (start..end).step_by(thread_chunk_size)
			.map(|lstart| (lstart, (lstart+thread_chunk_size).min(end)))
			.par_bridge()
			.map(|(lstart, lend)| {
				let unsafe_graph_ref = std::ptr::addr_of!(self.graph) as *mut RNNBuildGraph<R,F>;

				let mut new_neighbors: Vec<(F,usize,R,bool)> = vec![];
				let mut add_edges: Vec<(R,R,F)> = vec![];
				let mut remove_ids: Vec<usize> = vec![];

				(lstart..lend).map(|u| perm_gen.apply_rounds(u))
				.map(|u| (u,unsafe{R::from_usize(u).unwrap_unchecked()}))
				.for_each(|(_, u)| {
					let old_neighbors = unsafe{(*unsafe_graph_ref).get_adj_mut(u)};
					old_neighbors.sort_unstable_by(|(d1,_,_),(d2,_,_)| unsafe{d1.partial_cmp(d2).unwrap_unchecked()});
					new_neighbors.clear();
					remove_ids.clear();
					old_neighbors.into_iter().enumerate().for_each(|(i,&mut (vdist, v, vflag))| {
						let vusize = unsafe{R::to_usize(&v).unwrap_unchecked()};
						let mut keep_neighbor = true;
						for &(wdist, wusize, w, wflag) in new_neighbors.iter() {
							/* If both neighbors are "old" continue */
							if !vflag && !wflag { continue; }
							
							let vwdist: F = self._get_dist(mat, vusize, wusize);
							let cos = tri_to_cos(vdist,wdist,vwdist,dist_is_sq);
							if cos >= max_cos {
								// prune by RNG rule
								keep_neighbor = false;
								// insert connection w -> v instead:
								/* TODO: Verify sector exclusion criterion for new edge before insertion? */
								if !verify_sen_edges || vwdist < wdist || tri_to_cos(wdist,vwdist,vdist,dist_is_sq) < max_cos {
									add_edges.push((w, v, vwdist));
									break;
								}
							}
						}
						if keep_neighbor {
							new_neighbors.push((vdist,vusize,v,vflag));
						} else {
							remove_ids.push(i);
						}
					});
					remove_ids.iter().rev().for_each(|&i| _=old_neighbors.swap_remove(i));
					removed_edge_count.fetch_add(remove_ids.len(), Relaxed);
				});

				add_edges
			}).collect::<Vec<_>>();
			/* Collect all partial remove and add edges into one big vec, presort and apply */
			let n_add_edges = add_edges_list.iter().map(|ae| ae.len()).sum();
			let add_edges = &mut self.add_edge_cache;
			add_edges.clear();
			add_edges.reserve(n_add_edges);
			add_edges_list.into_iter().for_each(|ae| add_edges.extend(ae));
			/* Presort and remove duplicate insertions (duplicates in removals must stay) */
			unsafe {
				add_edges.par_sort_unstable_by(|(u1,v1, _),(u2,v2, _)| {
					let u_cmp = u1.partial_cmp(u2).unwrap_unchecked();
					if u_cmp.is_eq() { v1.partial_cmp(v2).unwrap_unchecked() } else { u_cmp }
				});
			}
			/* This is not important enough to justify single-thread work, we test for duplicate edges later on anyways */
			// remove_duplicates(add_edges);
			/* Apply edge changes in parallel and lock free */
			batch_set_flags(&mut self.graph, (start..end).map(|i|perm_gen.apply_rounds(i)).collect());
			batch_add_edges(&mut self.graph, true, add_edges);
		});
		/* Reduce the edge count to keep consistency */
		self.graph.n_edges -= removed_edge_count.load(Relaxed);
	}
	fn add_reverse_edges(&mut self) {
		/* Partition the workload into equal size for each thread */
		let thread_chunk_size = (self.n_data + current_num_threads() - 1) / current_num_threads();
		let n_data = self.n_data;
		let reduce_degree = self.params.reduce_degree;
		unsafe {
			/* Clear reverse edge cache */
			self.rev_edge_cache.par_chunks_mut(thread_chunk_size)
			.for_each(|chunk| {
				chunk.iter_mut().for_each(|rev_cache| rev_cache.clear());
			});
			/* Build reverse edge cache */
			(0..n_data).step_by(thread_chunk_size).par_bridge()
			.map(|start| (start, (start+thread_chunk_size).min(n_data)))
			.for_each(|(start, end)| {
				let mut neighbor_cache: Vec<(F,R,bool)> = Vec::with_capacity(2*reduce_degree);
				let unsafe_cache_ref = &mut neighbor_cache as *mut Vec<(F,R,bool)>;
				let unsafe_rev_cache_ref = std::ptr::addr_of!(self.rev_edge_cache) as *mut Vec<Vec<(F,R,bool)>>;
				(start..end).for_each(|uusize| {
					let u = R::from_usize(uusize).unwrap_unchecked();
					let old_neighbors = unsafe_cache_ref.as_mut().unwrap_unchecked();
					old_neighbors.clear();
					with_guard!(self, uusize, {
						let adj = self.graph.get_adj(u);
						old_neighbors.extend(adj);
						(&mut *unsafe_rev_cache_ref).get_unchecked_mut(uusize).extend(adj);
					});
					old_neighbors.iter().for_each(|&(dist, v, _)| {
						let vusize = R::to_usize(&v).unwrap_unchecked();
						with_guard!(self, vusize, (&mut *unsafe_rev_cache_ref).get_unchecked_mut(vusize).push((dist,u,true)));
					});
				})
			});
			/* Check and keep reverse edges */
			let n_total_edges = (0..n_data).step_by(thread_chunk_size).par_bridge()
			.map(|start| (start, (start+thread_chunk_size).min(n_data)))
			.map(|(start, end)| {
				let unsafe_graph_ref = std::ptr::addr_of!(self.graph) as *mut RNNBuildGraph<R,F>;
				let unsafe_rev_cache_ref = std::ptr::addr_of!(self.rev_edge_cache) as *mut Vec<Vec<(F,R,bool)>>;
				(start..end).map(|uusize| {
					let u = R::from_usize(uusize).unwrap_unchecked();
					let joined_neighbors = (&mut *unsafe_rev_cache_ref).get_unchecked_mut(uusize);
					/* Sorting such that: */
					/* - Closer neighbors are first */
					/* - Smaller IDs of equally far neighbors are first */
					/* - "not new" flag comes first if everything else is equal */
					joined_neighbors.sort_unstable_by(|(d1,v1,f1),(d2,v2,f2)| {
						let dist_cmp = d1.partial_cmp(d2).unwrap_unchecked();
						if dist_cmp.is_eq() {
							let id_cmp = v1.partial_cmp(v2).unwrap_unchecked();
							if id_cmp.is_eq() { f1.cmp(f2) } else { id_cmp }
						} else { dist_cmp }
					});
					remove_duplicates_with_key(joined_neighbors, |(_,v,_)| v);
					(*unsafe_graph_ref).clear_adj(u);
					let n_edges = reduce_degree.min(joined_neighbors.len());
					(*unsafe_graph_ref).get_adj_mut(u).extend(joined_neighbors[0..n_edges].iter());
					n_edges
				}).sum::<usize>()
			}).sum();
			self.graph.n_edges = n_total_edges;
		}
	}
	#[allow(unused)]
	fn prune_non_sen_edges<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let thread_chunk_size = self.params.concurrent_batch_size;
		let global_chunk_size = thread_chunk_size*current_num_threads();
		let (n_data, max_cos, dist_is_sq) = (self.n_data, self.params.max_cos, self.params.dist_is_sq);
		let perm_gen = RandomPermutationGenerator::new(n_data, 4);
		let removed_edge_count = AtomicUsize::new(0);
		(0..n_data).step_by(global_chunk_size)
		.map(|start| (start, (start+global_chunk_size).min(n_data)))
		.for_each(|(start, end)| {
			/* Compute edges to remove and to add in parallel */
			/* Working in chunks to remove the number of spawned threads and therefore overhead */
			(start..end).step_by(thread_chunk_size)
			.map(|lstart| (lstart, (lstart+thread_chunk_size).min(end)))
			.par_bridge()
			.for_each(|(lstart, lend)| {
				let unsafe_graph_ref = std::ptr::addr_of!(self.graph) as *mut RNNBuildGraph<R,F>;

				let mut new_neighbors: Vec<(F,usize,R,bool)> = vec![];
				let mut remove_ids: Vec<usize> = vec![];

				(lstart..lend).map(|u| perm_gen.apply_rounds(u))
				.map(|u| (u,unsafe{R::from_usize(u).unwrap_unchecked()}))
				.for_each(|(_, u)| {
					let old_neighbors = unsafe{(*unsafe_graph_ref).get_adj_mut(u)};
					old_neighbors.sort_unstable_by(|(d1,_,_),(d2,_,_)| unsafe{d1.partial_cmp(d2).unwrap_unchecked()});
					new_neighbors.clear();
					remove_ids.clear();
					old_neighbors.into_iter().enumerate().for_each(|(i,&mut (vdist, v, vflag))| {
						let vusize = unsafe{R::to_usize(&v).unwrap_unchecked()};
						let mut keep_neighbor = true;
						for &(wdist, wusize, _w, wflag) in new_neighbors.iter() {
							/* If both neighbors are "old" continue */
							if !vflag && !wflag { continue; }
							
							let vwdist: F = self._get_dist(mat, vusize, wusize);
							let cos = tri_to_cos(vdist,wdist,vwdist,dist_is_sq);
							if cos >= max_cos {
								// prune by RNG rule
								keep_neighbor = false;
								break;
							}
						}
						if keep_neighbor {
							new_neighbors.push((vdist,vusize,v,vflag));
						} else {
							remove_ids.push(i);
						}
					});
					remove_ids.iter().rev().for_each(|&i| _=old_neighbors.swap_remove(i));
					removed_edge_count.fetch_add(remove_ids.len(), Relaxed);
				});
			});
		});
		/* Reduce the edge count to keep consistency */
		self.graph.n_edges -= removed_edge_count.load(Relaxed);
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> RNNStyleBuilder<R, F, Dist> for SENDescentBuilder<R, F, Dist> {
	type Params = SENParams<F>;
	#[inline(always)]
	fn _mut_graph(&mut self) -> &mut RNNBuildGraph<R,F> { &mut self.graph }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _initial_degree(&self) -> usize { self.params.initial_degree }
	#[inline(always)]
	fn _into_graph_dist(self) -> (RNNBuildGraph<R,F>, Dist) { (self.graph, self.dist) }
	#[inline(always)]
	fn new<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let graph = RNNBuildGraph::new();
		let reduce_degree = params.reduce_degree;
		Self {
			_phantom: std::marker::PhantomData,
			n_data,
			params,
			node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
			rev_edge_cache: (0..n_data).map(|_| Vec::with_capacity(2*reduce_degree)).collect(),
			add_edge_cache: Vec::new(),
			graph,
			dist,
		}
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let (n_outer_loops, n_inner_loops) = (self.params.n_outer_loops, self.params.n_outer_loops);
		(0..n_outer_loops).for_each(|i| {
			(0..n_inner_loops).for_each(|_j| {
				self.update_neighbors(mat);
				#[cfg(debug_assertions)]
				println!("Outer loop: {:?}, Inner loop: {:?}, Nodes: {:?}, Edges: {:?}", i, _j, self.graph.n_vertices(), self.graph.n_edges());
			});
			if i != n_outer_loops-1 {
				self.add_reverse_edges();
				#[cfg(debug_assertions)]
				println!("Outer loop: {:?}, Inner loop: {:?}, Nodes: {:?}, Edges: {:?}", i, n_inner_loops, self.graph.n_vertices(), self.graph.n_edges());
			}
		});
		/* Sanity check to ensure the graph is simple directed */
		// self.remove_duplicate_edges();
		if self.params.prune_non_sen_edges { self.prune_non_sen_edges(mat); }
	}
}

param_struct!(SENEgoParams[Copy, Clone]<F: SyncFloat> {
	initial_degree: usize = 100,
	n_loops: usize = 4,
	concurrent_batch_size: usize = 100,
	max_cos: F = F::from(0.5).unwrap(),
	dist_is_sq: bool = true,
	radius: usize = 2,
});
impl<F: SyncFloat> SENEgoParams<F> {
	pub fn extend_rnn(params: RNNParams) -> Self {
		Self{
			initial_degree: params.initial_degree,
			n_loops: params.n_outer_loops * params.n_inner_loops,
			concurrent_batch_size: params.concurrent_batch_size,
			max_cos: F::from(0.5).unwrap(),
			dist_is_sq: true,
			radius: 2,
		}
	}
}
pub struct SENEgoDescentBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	params: SENEgoParams<F>,
	node_locks: Vec<Mutex<()>>,
	graph: RNNBuildGraph<R,F>,
	dist: Dist,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> SENEgoDescentBuilder<R, F, Dist> {
	fn update_neighbors<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let n_threads = current_num_threads();
		let thread_chunk_size = self.params.concurrent_batch_size;
		let global_chunk_size = thread_chunk_size*n_threads;
		let (n_data, max_cos, dist_is_sq, radius) = (self.n_data, self.params.max_cos, self.params.dist_is_sq, self.params.radius);
		let perm_gen = RandomPermutationGenerator::new(n_data, 4);
		(0..n_data).step_by(global_chunk_size)
		.map(|start| (start, (start+global_chunk_size).min(n_data)))
		.for_each(|(start, end)| {
			let add_edge_count = AtomicUsize::new(0);
			let remove_edge_count = AtomicUsize::new(0);
			/* Compute edges to remove and to add in parallel */
			/* Working in chunks to remove the number of spawned threads and therefore overhead */
			(start..end).step_by(thread_chunk_size)
			.map(|lstart| (lstart, (lstart+thread_chunk_size).min(end)))
			.par_bridge()
			.for_each(|(lstart, lend)| {
				let unsafe_graph_ref = std::ptr::addr_of!(self.graph) as *mut RNNBuildGraph<R,F>;

				let mut kept_neighbor_cache: Vec<(F,R,bool)> = vec![];
				let mut visited: HashSet<R> = HashSet::default();
				let mut work_queue: Vec<R> = Vec::new();
				let mut next_work_queue: Vec<R> = Vec::new();
				let mut found_ids = Vec::new();

				(lstart..lend).map(|u| perm_gen.apply_rounds(u))
				.map(|u| (u,unsafe{R::from_usize(u).unwrap_unchecked()}))
				.for_each(|(uusize, u)| {
					visited.clear();
					work_queue.clear();
					next_work_queue.clear();
					found_ids.clear();
					with_guard!(self, uusize, {
						let graph_ref = unsafe{&mut *unsafe_graph_ref};
						visited.insert(u);
						work_queue.push(u);
						for _ in 0..radius {
							next_work_queue.clear();
							work_queue.iter().for_each(|&v| {
								graph_ref.get_adj(v).iter().for_each(|&(_,n,_)| {
									if !visited.contains(&n) {
										visited.insert(n);
										next_work_queue.push(n);
										found_ids.push(n);
									}
								});
							});
							swap(&mut work_queue, &mut next_work_queue);
						}
					});
					let mut u_neighbor_cands: Vec<(usize, R, F)> = found_ids.iter()
					.map(|&v| {
						let vusize = unsafe{R::to_usize(&v).unwrap_unchecked()};
						let uvdist = self._get_dist(mat, uusize,vusize);
						(vusize, v, uvdist)
					}).collect();
					u_neighbor_cands.sort_unstable_by(|(_,_,d1),(_,_,d2)| d1.partial_cmp(d2).unwrap());
					kept_neighbor_cache.clear();
					u_neighbor_cands.iter().for_each(|&(vusize, v, vdist)| {
						let mut keep_neighbor = true;
						for &(wdist, w, _) in &kept_neighbor_cache {
							let wusize = unsafe{R::to_usize(&w).unwrap_unchecked()};
							let vwdist = self._get_dist(mat, vusize,wusize);
							let cos = tri_to_cos(vdist,wdist,vwdist,dist_is_sq);
							if cos >= max_cos {
								// prune by RNG rule
								keep_neighbor = false;
								break;
							}
						}
						if keep_neighbor {
							kept_neighbor_cache.push((vdist, v, true));
						}
					});
					with_guard!(self, uusize, {
						let adj = unsafe{(*unsafe_graph_ref).get_adj_mut(u)};
						remove_edge_count.fetch_add(adj.len(), Relaxed);
						swap(adj, &mut kept_neighbor_cache);
						add_edge_count.fetch_add(adj.len(), Relaxed);
					});
				});
			});
			/* Update the edge count to keep consistency */
			// println!("Remove edges: {:?}, Add edges: {:?}", remove_edge_count.load(Relaxed), add_edge_count.load(Relaxed));
			self.graph.n_edges -= remove_edge_count.load(Relaxed);
			self.graph.n_edges += add_edge_count.load(Relaxed);
		});
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> RNNStyleBuilder<R, F, Dist> for SENEgoDescentBuilder<R, F, Dist> {
	type Params = SENEgoParams<F>;
	#[inline(always)]
	fn _mut_graph(&mut self) -> &mut RNNBuildGraph<R,F> { &mut self.graph }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _initial_degree(&self) -> usize { self.params.initial_degree }
	#[inline(always)]
	fn _into_graph_dist(self) -> (RNNBuildGraph<R,F>, Dist) { (self.graph, self.dist) }
	fn new<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let graph = RNNBuildGraph::new();
		Self {
			_phantom: std::marker::PhantomData,
			n_data,
			params,
			node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
			graph,
			dist,
		}
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		assert_eq!(self.graph.n_edges(), (0..self.graph.n_vertices()).map(|u| self.graph.get_adj(R::from_usize(u).unwrap()).len()).sum::<usize>());
		println!("After initialization, Nodes: {:?}, Edges: {:?}", self.graph.n_vertices(), self.graph.n_edges());
		let true_radius = self.params.radius;
		(0..self.params.n_loops).for_each(|_i| {
			if _i > 0 {
				let mut add_edges = (0..self.graph.n_vertices()).map(|u| {
					let u = R::from_usize(u).unwrap();
					self.graph.get_adj(u).iter()
					.map(move |&(dist,v,_)| (v,u,dist))
				}).flatten().collect();
				batch_add_edges(&mut self.graph, false, &mut add_edges);
			}
			self.params.radius = 1;
			self.update_neighbors(mat);
			let n_rand_edges = self.graph.n_edges()/2;
			add_random_edges(mat, &mut self.graph, &self.dist, n_rand_edges);
			self.params.radius = true_radius;
			self.update_neighbors(mat);
			// #[cfg(debug_assertions)]
			println!("Loop: {:?}, Nodes: {:?}, Edges: {:?}", _i, self.graph.n_vertices(), self.graph.n_edges());
		});
		/* Sanity check to ensure the graph is simple directed */
		// self.remove_duplicate_edges();
	}
}


pub fn bruteforce_sen_edges<F: SyncFloat, M: MatrixDataSource<F>+Sync>(data: &M, max_cos: F) -> Vec<(usize, usize)> {
	use ndarray::Array2;
	use graphidx::measures::SquaredEuclideanDistance;
	let mut result = Vec::new();
	let n_data = data.n_rows();
	let n_threads = current_num_threads();
	let chunk_size = (n_data+n_threads-1)/n_threads;
	let dist = SquaredEuclideanDistance::new();
	/* Bruteforce all pairwise distances of the dataset */
	let dist_mat = Array2::from_elem((n_data,n_data), F::zero());
	let perm_gen = RandomPermutationGenerator::new(n_data, 4);
	(0..n_data).step_by(chunk_size)
	.map(|start| (start, (start+chunk_size).min(n_data)))
	.par_bridge()
	.for_each(|(start, end)| {
		let unsafe_dist_mat_ref = std::ptr::addr_of!(dist_mat) as *mut Array2<F>;
		(start..end).for_each(|i| {
			let i = perm_gen.apply_rounds(i);
			let p = data.get_row_view(i);
			(0..i).for_each(|j: usize| {
				let q = data.get_row_view(j);
				let dist = dist.dist_slice(&p, &q);
				unsafe {
					(&mut *unsafe_dist_mat_ref)[[i,j]] = dist;
					(&mut *unsafe_dist_mat_ref)[[j,i]] = dist;
				}
			});
		});
	});
	let mut dist_cache = vec![(0, F::zero()); n_data];
	let mut edge_cache: Vec<usize> = Vec::new();
	(0..n_data).for_each(|i| {
		/* Bruteforce compute all distances */
		(0..n_data).step_by(chunk_size)
		.map(|start| (start, (start+chunk_size).min(n_data)))
		.zip(dist_cache.chunks_mut(chunk_size))
		.par_bridge()
		.for_each(|((start, end), dist_chunk)| {
			(start..end).for_each(|j| dist_chunk[j-start] = (j, dist_mat[[i,j]]));
		});
		dist_cache.par_sort_unstable_by(|(_,d1),(_,d2)| d1.partial_cmp(d2).unwrap());
		/* Find edges by RNG rule */
		edge_cache.clear();
		dist_cache.iter().skip(1).for_each(|&(j,_)| {
			if edge_cache.iter().all(|&k| {
				let (ij2,ik2,jk2) = (dist_mat[[i,j]], dist_mat[[i,k]], dist_mat[[j,k]]);
				let ijik = <F as num::Float>::sqrt((ij2*ik2).max(F::zero()));
				if ijik <= F::zero() { true } else {
					let cos = (ij2+ik2-jk2)/(ijik+ijik);
					cos < max_cos
				}
			}) {
				edge_cache.push(j);
			}
		});
		result.extend(edge_cache.iter().map(|&j| (i,j)));
	});
	result
}


#[cfg(test)]
mod tests {
	use crate::rnn::*;
	use graphidx::{indices::*, measures::EuclideanDistance};
	use ndarray::{Array2, Slice};
	use ndarray_rand::rand_distr::{Uniform,Normal};
	use rand::prelude::Distribution;

	#[test]
	fn par_iter_chunk_test () {
		use rand::Rng;
		use rayon::prelude::ParallelSliceMut;
		use rayon::prelude::ParallelSlice;
		use rayon::iter::ParallelIterator;
		use rayon::iter::IntoParallelIterator;
		use std::collections::HashSet;
		let mut rng = rand::thread_rng();
		let mut a: Vec<(usize,usize)> = (0..100_000).map(|_| (rng.gen_range(0..100), rng.gen_range(0..100))).collect::<Vec<_>>();
		a.par_sort_unstable_by(|(u1,_),(u2,_)| unsafe{u1.partial_cmp(u2).unwrap_unchecked()});
		let b = a.par_chunk_by(|(u1,_),(u2,_)| u1==u2).into_par_iter().map(|chunk| chunk[0].0).collect::<Vec<_>>();
		let c = b.iter().map(|&x|x).collect::<HashSet<_>>();
		assert_eq!(b.len(), c.len());

		use num::FromPrimitive;
		use num::ToPrimitive;
		assert!(
			(0..1_000_000)
			.map(|_| rng.gen_range(0..10_000))
			.map(|x: usize| (<usize as FromPrimitive>::from_usize(x).unwrap().to_usize().unwrap(),x))
			.all(|(x,y)|x==y)
		);
	}

	#[test]
	fn relative_nn_construction() {
		use ndarray::Array2;
		let (n,d) = (10_000, 20);
		let (deg_init, deg_rec, n_outer, n_inner) = (30, 30, 5, 5);
		let rng = Normal::new(0.0, 1.0).unwrap();
		let data: Array2<f64> = Array2::from_shape_fn((n, d), |_| rng.sample(&mut rand::thread_rng()));
		let params = RNNParams::new()
		.with_initial_degree(deg_init)
		.with_reduce_degree(deg_rec)
		.with_n_outer_loops(n_outer)
		.with_n_inner_loops(n_inner);
		let _graph = RNNDescentBuilder::<u64,_,_>::build(data, EuclideanDistance::new(), params);
	}
	
	#[test]
	fn relative_nn_query() {
		/* Imports */
		use ndarray::Axis;
		use graphidx::measures::*;
		use graphidx::graphs::Graph;
		use std::time::Instant;
		/* Limit global thread pool size */
		// rayon::ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();
		/* Parameters */
		let (nd, nq, d, k) = (100_000, 1000, 50, 10);
		let bruteforce_edges = false;
		let euc = SquaredEuclideanDistance::new();
		let max_edges = d;
		/* Data initialization */
		let init_time = Instant::now();
		type R = usize;
		type F = f32;
		let data: Array2<F> = if 0>0 { /* Normal distribution */
			let rng = Normal::new(0.0, 1.0).unwrap();
			Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()))
		} else { /* Uniform distribution */
			let rng = Uniform::new(0.0, 1.0);
			Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()))
		};
		let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
		println!("Data initialization: {:.2?}", init_time.elapsed());
		/* Build and translate RNN Graph */
		let build_time = Instant::now();
		let dist = SquaredEuclideanDistance::new();
		let rnn_base_params = RNNParams::new()
		.with_initial_degree(2*max_edges)
		.with_reduce_degree(max_edges)
		.with_n_outer_loops(4)
		.with_n_inner_loops(15);
		#[allow(unused)]
		enum IdxType {
			RNG,
			RNGEgo,
			SEN,
			SENEgo,
		}
		let rnn_type = IdxType::SEN;
		let index1 = match rnn_type {
			IdxType::RNG => { /* RNG Descent */
				let ret = RNNDescentBuilder::<R,_,_>::build(data.view(), dist, rnn_base_params);
				println!("Graph construction: {:.2?}", build_time.elapsed());
				println!("Average out degree: {:.2?}", ret.n_edges() as F / (nd as F));
				if bruteforce_edges {
					/* Brute force RNG edges */
					let rng_time = Instant::now();
					let rng_edges = bruteforce_rng_edges(&data);
					println!("Brute force RNG edges: {:.2?}", rng_time.elapsed());
					/* Check RNG edges */
					let n_rng = rng_edges.len();
					let n_graph = ret.n_edges();
					let contained = rng_edges.iter().map(|&(i,j)| ret.graph().find_edge(i, j).is_some() as usize).sum::<usize>();
					println!("RNG edges are in graph: {:.2}%", contained as F / n_rng as F * 100.0);
					println!("Edges in graph are RNG: {:.2}%", contained as F / n_graph as F * 100.0);
				}
				ret
			},
			IdxType::RNGEgo => { /* RNG Ego Descent */
				let rnn_ego_params = RNNEgoParams::extend_rnn(rnn_base_params)
				.with_initial_degree(10)
				.with_n_loops(10)
				.with_concurrent_batch_size(10)
				.with_radius(2)
				;
				let ret = RNNEgoDescentBuilder::<R,_,_>::build(data.view(), dist, rnn_ego_params);
				println!("Graph construction: {:.2?}", build_time.elapsed());
				println!("Average out degree: {:.2?}", ret.n_edges() as F / (nd as F));
				if bruteforce_edges {
					/* Brute force RNG edges */
					let rng_time = Instant::now();
					let rng_edges = bruteforce_rng_edges(&data);
					println!("Brute force RNG edges: {:.2?}", rng_time.elapsed());
					/* Check RNG edges */
					let n_rng = rng_edges.len();
					let n_graph = ret.n_edges();
					let contained = rng_edges.iter().map(|&(i,j)| ret.graph().find_edge(i, j).is_some() as usize).sum::<usize>();
					println!("RNG edges are in graph: {:.2}%", contained as F / n_rng as F * 100.0);
					println!("Edges in graph are RNG: {:.2}%", contained as F / n_graph as F * 100.0);
				}
				ret
			},
			IdxType::SEN => { /* SEN */
				let max_cos = 0.5 as F;
				// let max_cos = 0.35958121131989174 as F
				// let max_cos = 0.6433347146667074 as F;
				let sen_params = SENParams::extend_rnn(rnn_base_params)
				.with_max_cos(max_cos)
				.with_dist_is_sq(true)
				.with_prune_non_sen_edges(false)
				.with_verify_sen_edges(false)
				// .with_n_outer_loops(10)
				// .with_n_inner_loops(1500)
				// .with_initial_degree(6*max_edges)
				// .with_reduce_degree(10*max_edges)
				// .with_initial_degree(2*254)
				// .with_reduce_degree(254)
				;
				let ret = SENDescentBuilder::<R,_,_>::build(data.view(), SquaredEuclideanDistance::new(), sen_params);
				println!("Graph construction: {:.2?}", build_time.elapsed());
				println!("Average out degree: {:.2?}", ret.n_edges() as F / (nd as F));
				if bruteforce_edges {
					/* Brute force RNG edges */
					let sen_time = Instant::now();
					let sen_edges = bruteforce_sen_edges(&data, max_cos);
					println!("Brute force SEN edges: {:.2?}", sen_time.elapsed());
					// println!("SEN edges: {:?}", sen_edges.len());
					/* Check RNG edges */
					let n_sen = sen_edges.len();
					let n_graph = ret.n_edges();
					let contained = sen_edges.iter().map(|&(i,j)| ret.graph().find_edge(i, j).is_some() as usize).sum::<usize>();
					println!("SEN edges are in graph: {:.2}%", contained as F / n_sen as F * 100.0);
					println!("Edges in graph are SEN: {:.2}%", contained as F / n_graph as F * 100.0);
				}
				ret
			},
			IdxType::SENEgo => { /* SEN Ego Descent */
				let max_cos = 0.4 as F;
				let sen_params = SENEgoParams::extend_rnn(rnn_base_params)
				.with_initial_degree(100)
				.with_n_loops(50)
				.with_max_cos(max_cos)
				.with_dist_is_sq(true)
				.with_concurrent_batch_size(100)
				.with_radius(2)
				;
				let ret = SENEgoDescentBuilder::<R,_,_>::build(data.view(), SquaredEuclideanDistance::new(), sen_params);
				println!("Graph construction: {:.2?}", build_time.elapsed());
				println!("Average out degree: {:.2?}", ret.n_edges() as F / (nd as F));
				if bruteforce_edges {
					/* Brute force RNG edges */
					let sen_time = Instant::now();
					let sen_edges = bruteforce_sen_edges(&data, max_cos);
					println!("Brute force SEN edges: {:.2?}", sen_time.elapsed());
					// println!("SEN edges: {:?}", sen_edges.len());
					/* Check RNG edges */
					let n_sen = sen_edges.len();
					let n_graph = ret.n_edges();
					let contained = sen_edges.iter().map(|&(i,j)| ret.graph().find_edge(i, j).is_some() as usize).sum::<usize>();
					println!("SEN edges are in graph: {:.2}%", contained as F / n_sen as F * 100.0);
					println!("Edges in graph are SEN: {:.2}%", contained as F / n_graph as F * 100.0);
				}
				ret
			},
		};
		assert!((0..nd).all(|i| {
				let neighbors = index1.graph().neighbors(i);
				neighbors.iter().collect::<HashSet<_>>().len() == neighbors.len()
		}));
		let index2 = GreedyCappedSingleGraphIndex::new(
			data.view(),
			index1.graph().as_dir_lol_graph(),
			SquaredEuclideanDistance::new(),
			3*k,
			None,
		);
		/* Brute force queries */
		let bruteforce_time = Instant::now();
		let (bruteforce_ids, _) = bruteforce_neighbors(&data, &queries, &euc, k);
		println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());
		// println!("{:?}", bruteforce_ids.row(0));
		// println!("{:?}", bruteforce_dists.row(0));
		/* RNN queries */
		let rnn_time = Instant::now();
		#[allow(unused)]
		let (rnn_ids1, rnn_dists1) = index1.greedy_search_batch(&queries, k, 10*k);
		println!("RNN queries 1: {:.2?}", rnn_time.elapsed());
		// println!("{:?}", rnn_ids1.row(0));
		// println!("{:?}", rnn_dists1.row(0));
		let rnn_time = Instant::now();
		#[allow(unused)]
		let (rnn_ids2, rnn_dists2) = index2.greedy_search_batch(&queries, k, 10*k);
		println!("RNN queries 2: {:.2?}", rnn_time.elapsed());
		// println!("{:?}", rnn_ids2.row(0));
		// println!("{:?}", rnn_dists2.row(0));
		/* Compute and print recall */
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(rnn_ids1.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let rnn_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&rnn_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 1: {:.2}%", recall*100f32);
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(rnn_ids2.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let rnn_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&rnn_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 2: {:.2}%", recall*100f32);
	}
}
