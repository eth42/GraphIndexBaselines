use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Mutex;

use graphidx::bit_vectors::BitVector;
use graphidx::bit_vectors::BitVectorMut;
use graphidx::graphs::*;
use graphidx::heaps::*;
use graphidx::random::RandomPermutationGenerator;
use graphidx::types::*;
use graphidx::indices::*;
use graphidx::measures::*;
use graphidx::data::*;
use graphidx::sets::{HashSetLike,HashOrBitset};
use rayon::current_num_threads;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use ndarray::Array2;

use crate::util::remove_duplicates_with_key;
use crate::rnn::{RNNStyleBuilder, RNNParams, RNNDescentBuilder, SENParams, SENDescentBuilder};

// type HashSet<T> = foldhash::HashSet<T>;
// type HashSet<T> = graphidx::sets::ArraySet<T,u8>;
type HashSet<T> = graphidx::sets::BitSet<T>;
// type HashSet<T> = graphidx::sets::ApproxBitSet<T>;
// type HashSet<T> = graphidx::sets::TrackingBitSet<T>;
// type HashSet<T> = graphidx::sets::SwappingArraySet<T,u8>;
// type HashSet<T> = graphidx::sets::NAryArraySet<T,u8>;



type HNSWBuildGraph<R,F> = WDirLoLGraph<R,F>;

pub struct HNSWHeapBuildGraph<R: UnsignedInteger, F: Float> {
	adjacency: Vec<MaxHeap<F,R>>,
	n_edges: usize
}
impl<R: UnsignedInteger, F: Float> HNSWHeapBuildGraph<R,F> {
	#[inline(always)]
	pub fn new() -> Self {
		Self{adjacency: vec![], n_edges:0}
	}
	#[inline(always)]
	pub fn view_neighbors_heap(&self, vertex: R) -> &MaxHeap<F,R> {
		&self.adjacency[vertex.to_usize().unwrap()]
	}
	#[inline(always)]
	pub fn view_neighbors_heap_mut(&mut self, vertex: R) -> &mut MaxHeap<F,R> {
		&mut self.adjacency[vertex.to_usize().unwrap()]
	}
}
impl<R: UnsignedInteger, F: Float> Graph<R> for HNSWHeapBuildGraph<R,F> {
	#[inline(always)]
	fn clear_neighbors(&mut self, vertex: R) {
		self.view_neighbors_heap_mut(vertex).clear();
	}
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
		self.adjacency[vertex.to_usize().unwrap()].iter().map(|&(_,v)| v).collect()
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
		self.adjacency.push(MaxHeap::new());
	}
	#[inline(always)]
	fn add_node_with_capacity(&mut self, capacity: usize) {
		self.adjacency.push(MaxHeap::with_capacity(capacity));
	}
	#[inline(always)]
	fn add_edge(&mut self, _vertex1: R, _vertex2: R) {
		panic!("Cannot add edge without weight to a weighted graph");
	}
	#[inline(always)]
	fn remove_edge_by_index(&mut self, _vertex: R, _index: usize) {
		panic!("Cannot remove edge by index in heap-based graph");
	}
}
impl<R: UnsignedInteger, F: Float> WeightedGraph<R,F> for HNSWHeapBuildGraph<R,F> {
	#[inline(always)]
	fn edge_weight(&self, vertex1: R, vertex2: R) -> F {
		self.adjacency[vertex1.to_usize().unwrap()].iter().find(|&&(_,v)| v == vertex2).unwrap().0
	}
	#[inline(always)]
	fn add_edge_with_weight(&mut self, vertex1: R, vertex2: R, weight: F) {
		self.adjacency[vertex1.to_usize().unwrap()].push(weight, vertex2);
		self.n_edges += 1;
	}
	#[inline(always)]
	fn neighbors_with_weights(&self, vertex: R) -> (Vec<F>, Vec<R>) {
		let mut neighbors = Vec::new();
		let mut weights = Vec::new();
		for &(w,v) in self.adjacency[vertex.to_usize().unwrap()].iter() {
			neighbors.push(v);
			weights.push(w);
		}
		(weights, neighbors)
	}
	#[inline(always)]
	fn neighbors_with_zipped_weights(&self, vertex: R) -> Vec<(F,R)> {
		self.adjacency[vertex.to_usize().unwrap()].iter().map(|&v|v).collect()
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight<Fun: FnMut(&F, &R)>(&self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter().for_each(|&v| f(&v.0,&v.1));
	}
	#[inline(always)]
	fn foreach_neighbor_with_zipped_weight_mut<Fun: FnMut(&mut F, &mut R)>(&mut self, vertex: R, mut f: Fun) {
		self.adjacency[vertex.to_usize().unwrap()].iter_mut().for_each(|(w,v)| f(w,v));
	}
	#[inline(always)]
	fn as_viewable_weighted_adj_graph(&self) -> Option<&impl ViewableWeightedAdjGraph<R,F>> {
		Some(self)
	}
}
impl<R: UnsignedInteger, F: Float> ViewableWeightedAdjGraph<R,F> for HNSWHeapBuildGraph<R,F> {
	#[inline(always)]
	fn view_neighbors(&self, vertex: R) -> &[(F,R)] {
		self.adjacency[vertex.to_usize().unwrap()].as_slice()
	}
	#[inline(always)]
	fn view_neighbors_mut(&mut self, vertex: R) -> &mut [(F,R)] {
		self.adjacency[vertex.to_usize().unwrap()].as_mut_slice()
	}
}



#[inline(always)]
fn make_greedy_index<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	M: MatrixDataSource<F>,
	Dist: Distance<F>,
	G: Graph<R>,
>(graphs: Vec<G>, local_layer_ids: Vec<Vec<R>>, global_layer_ids: Vec<Vec<R>>, mat: M, dist: Dist, higher_level_max_heap_size: usize) -> GreedyLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
	GreedyLayeredGraphIndex::new(
		mat,
		graphs.iter().map(|g|g.as_dir_lol_graph()).collect(),
		local_layer_ids,
		global_layer_ids,
		dist,
		higher_level_max_heap_size,
		Some(vec![R::zero()]),
	)
}
#[inline(always)]
fn make_greedy_capped_index<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	M: MatrixDataSource<F>,
	Dist: Distance<F>,
	G: Graph<R>,
>(graphs: Vec<G>, local_layer_ids: Vec<Vec<R>>, global_layer_ids: Vec<Vec<R>>, mat: M, dist: Dist, higher_level_max_heap_size: usize, max_frontier_size: usize) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
	GreedyCappedLayeredGraphIndex::new(
		mat,
		graphs.iter().map(|g|g.as_dir_lol_graph()).collect(),
		local_layer_ids,
		global_layer_ids,
		dist,
		higher_level_max_heap_size,
		max_frontier_size, Some(vec![R::zero()]),
	)
}
#[inline(always)]
fn make_fat_greedy_index<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	M: MatrixDataSource<F>,
	Dist: Distance<F>,
	G: Graph<R>,
>(graphs: Vec<G>, global_layer_ids: Vec<Vec<R>>, mat: M, dist: Dist, higher_level_max_heap_size: usize, higher_level_max_degree: usize, lowest_level_max_degree: usize) -> GreedyLayeredGraphIndex<R, F, Dist, M, FatDirGraph<R>> {
	let top_entry_points = if global_layer_ids.len() > 0 { Some(vec![global_layer_ids.last().unwrap()[0]]) } else { Some(vec![R::zero()]) };
	let mut fat_graphs = Vec::new();
	fat_graphs.push(graphs[0].as_fat_dir_graph(
		None,
		Some(mat.n_rows()),
		Some(lowest_level_max_degree))
	);
	graphs.iter().skip(1)
	.zip(global_layer_ids.into_iter())
	.for_each(|(g, global_ids)| {
		fat_graphs.push(g.as_fat_dir_graph(
			Some(global_ids),
			Some(mat.n_rows()),
			Some(higher_level_max_degree)
		))
	});
	GreedyLayeredGraphIndex::new(
		mat,
		fat_graphs,
		Vec::new(),
		Vec::new(),
		dist,
		higher_level_max_heap_size,
		top_entry_points,
	)
}
#[inline(always)]
fn make_fat_greedy_capped_index<
	R: SyncUnsignedInteger,
	F: SyncFloat,
	M: MatrixDataSource<F>,
	Dist: Distance<F>,
	G: Graph<R>,
>(graphs: Vec<G>, global_layer_ids: Vec<Vec<R>>, mat: M, dist: Dist, higher_level_max_heap_size: usize, higher_level_max_degree: usize, lowest_level_max_degree: usize, max_frontier_size: usize) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, FatDirGraph<R>> {
	let top_entry_points = if global_layer_ids.len() > 0 { Some(vec![global_layer_ids.last().unwrap()[0]]) } else { Some(vec![R::zero()]) };
	let mut fat_graphs = Vec::new();
	fat_graphs.push(graphs[0].as_fat_dir_graph(
		None,
		Some(mat.n_rows()),
		Some(lowest_level_max_degree))
	);
	graphs.iter().skip(1)
	.zip(global_layer_ids.into_iter())
	.for_each(|(g, global_ids)| {
		fat_graphs.push(g.as_fat_dir_graph(
			Some(global_ids),
			Some(mat.n_rows()),
			Some(higher_level_max_degree))
		);
	});
	GreedyCappedLayeredGraphIndex::new(
		mat,
		fat_graphs,
		Vec::new(),
		Vec::new(),
		dist,
		higher_level_max_heap_size,
		max_frontier_size,
		top_entry_points,
	)
}

#[inline(always)]
pub fn random_level(level_norm_param: f32, max_level: usize) -> usize {
    let f = rand::random::<f32>();
    let level = (-f.ln() * level_norm_param).floor() as usize;
    level.min(max_level - 1)
}




pub trait HNSWStyleBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> where Self: Sized {
	type Params;
	type Graph: ViewableWeightedAdjGraph<R,F>;
	fn _graphs(&self) -> &Vec<Self::Graph>;
	fn _mut_graphs(&mut self) -> &mut Vec<Self::Graph>;
	fn _global_layer_ids(&self) -> &Vec<Vec<R>>;
	fn _dist(&self) -> &Dist;
	fn _into_parts(self) -> (Vec<Self::Graph>, Vec<Vec<R>>, Vec<Vec<R>>, Dist);
	fn _max_build_heap_size(&self) -> usize;
	fn _max_build_frontier_size(&self) -> Option<usize>;
	fn _max_degrees(&self) -> (usize, usize);
	#[inline(always)]
	fn _get_dist<M: MatrixDataSource<F>>(&self, mat: &M, i: usize, j: usize) -> F {
		if M::SUPPORTS_ROW_VIEW {
			self._dist().dist_slice(mat.get_row_view(i), mat.get_row_view(j))
		} else {
			self._dist().dist(&mat.get_row(i), &mat.get_row(j))
		}
	}
	fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self;
	#[inline(always)]
	fn build<M: MatrixDataSource<F>+Sync>(mat: M, dist: Dist, params: Self::Params, higher_level_max_heap_size: usize) -> GreedyLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
		let builder = Self::base_init(&mat, dist, params);
		let (graphs, local_layer_ids, global_layer_ids, dist) = builder._into_parts();
		make_greedy_index(graphs, local_layer_ids, global_layer_ids, mat, dist, higher_level_max_heap_size)
	}
	#[inline(always)]
	fn build_capped<M: MatrixDataSource<F>+Sync>(mat: M, dist: Dist, params: Self::Params, higher_level_max_heap_size: usize, max_frontier_size: usize) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, DirLoLGraph<R>> {
		let builder = Self::base_init(&mat, dist, params);
		let (graphs, local_layer_ids, global_layer_ids, dist) = builder._into_parts();
		make_greedy_capped_index(graphs, local_layer_ids, global_layer_ids, mat, dist, higher_level_max_heap_size, max_frontier_size)
	}
	#[inline(always)]
	fn build_fat<M: MatrixDataSource<F>+Sync>(mat: M, dist: Dist, params: Self::Params, higher_level_max_heap_size: usize) -> GreedyLayeredGraphIndex<R, F, Dist, M, FatDirGraph<R>> {
		let builder = Self::base_init(&mat, dist, params);
		let (lowest_max_degree, higher_max_degree) = builder._max_degrees();
		let (graphs, _, global_layer_ids, dist) = builder._into_parts();
		make_fat_greedy_index(graphs, global_layer_ids, mat, dist, higher_level_max_heap_size, higher_max_degree, lowest_max_degree)
	}
	#[inline(always)]
	fn build_fat_capped<M: MatrixDataSource<F>+Sync>(mat: M, dist: Dist, params: Self::Params, higher_level_max_heap_size: usize, max_frontier_size: usize) -> GreedyCappedLayeredGraphIndex<R, F, Dist, M, FatDirGraph<R>> {
		let builder = Self::base_init(&mat, dist, params);
		let (lowest_max_degree, higher_max_degree) = builder._max_degrees();
		let (graphs, _, global_layer_ids, dist) = builder._into_parts();
		make_fat_greedy_capped_index(graphs, global_layer_ids, mat, dist, higher_level_max_heap_size, higher_max_degree, lowest_max_degree, max_frontier_size)
	}
	/// Searches in the specified layer.
	/// 
	/// Assertions:
	/// - i is the **global ID** of the point to insert
	/// - entry_points is populated with the desired entry points
	/// - all other caches can be wiped
	/// 
	/// Result:
	/// - entry_points is populated with up to max_build_heap_size (or max_heap_size_override) neighbors in ascending distance
	/// - state of all other caches is undefined
	fn _search_layer<M: MatrixDataSource<F>+Sync>(&self, mat: &M, i: usize, layer: usize, visited_set: &mut impl HashSetLike<R>, search_maxheap: &mut MaxHeap<F,R>, frontier_minheap: &mut MinHeap<F,R>, frontier_dualheap: &mut DualHeap<F,R>, entry_points: &mut Vec<(F,R)>, max_heap_size_override: Option<usize>) {
		let graph = &self._graphs()[layer];
		visited_set.clear();
		let max_heap_size = max_heap_size_override.unwrap_or(if layer==0 {self._max_build_heap_size()} else {1});
		let is_layer0 = layer == 0;
		if max_heap_size > 1 {
			search_maxheap.clear();
			/* Populate heap with entry points */
			entry_points.into_iter().for_each(|&mut (d,i)| {
				if search_maxheap.size() < max_heap_size {
					search_maxheap.push(d, i);
				} else {
					search_maxheap.push_pop(d, i);
				}
				visited_set.insert(i);
			});
			if self._max_build_frontier_size().is_none() {
				frontier_minheap.clear();
				entry_points.into_iter().for_each(|&mut (d,i)| frontier_minheap.push(d,i));
				let global_ids = if is_layer0 {None} else {Some(&self._global_layer_ids()[layer-1])};
				while let Some((d, v)) = frontier_minheap.pop() {
					if d > search_maxheap.peek().unwrap().0 { break; }
					for &(_,j) in graph.view_neighbors(v) {
						if visited_set.insert(j) {
							let j_global = unsafe{(if is_layer0 { j } else { global_ids.unwrap_unchecked()[j.to_usize().unwrap_unchecked()] }).to_usize().unwrap_unchecked()};
							let neighbor_dist = self._get_dist(mat, i, j_global);
							if search_maxheap.size() < max_heap_size {
								search_maxheap.push(neighbor_dist, j);
							} else {
								search_maxheap.push_pop(neighbor_dist, j);
							}
							frontier_minheap.push(neighbor_dist, j);
						}
					}
				}
			} else {
				frontier_dualheap.clear();
				let max_frontier_size = unsafe{self._max_build_frontier_size().unwrap_unchecked()};
				entry_points.into_iter().for_each(|&mut (d,i)| {
					if frontier_dualheap.size() < max_frontier_size {
						frontier_dualheap.push(d,i);
					} else {
						frontier_dualheap.push_pop::<false>(d,i);
					}
				});
				let global_ids = if is_layer0 {None} else {Some(&self._global_layer_ids()[layer-1])};
				while let Some((d, v)) = frontier_dualheap.pop::<true>() {
					if d > search_maxheap.peek().unwrap().0 { break; }
					for &(_,j) in graph.view_neighbors(v) {
						if visited_set.insert(j) {
							let j_global = unsafe{(if is_layer0 { j } else { global_ids.unwrap_unchecked()[j.to_usize().unwrap_unchecked()] }).to_usize().unwrap_unchecked()};
							let neighbor_dist = self._get_dist(mat, i, j_global);
							if search_maxheap.size() < max_heap_size {
								search_maxheap.push(neighbor_dist, j);
							} else {
								search_maxheap.push_pop(neighbor_dist, j);
							}
							if frontier_dualheap.size() < max_frontier_size {
								frontier_dualheap.push(neighbor_dist, j);
							} else {
								frontier_dualheap.push_pop::<false>(neighbor_dist, j);
							}
						}
					}
				}
			}
			/* Since we have a max heap, we have to reverse the order to get an ascending distance list */
			entry_points.clear();
			entry_points.reserve(search_maxheap.size());
			unsafe{entry_points.set_len(search_maxheap.size());}
			entry_points.iter_mut().rev().zip(search_maxheap.sorted_iter()).for_each(|(x, y)| *x = y);
		} else {
			let (mut min_dist, mut min_idx) = (F::infinity(), R::zero());
			/* Populate heap with entry points */
			entry_points.into_iter().for_each(|&mut (d,i)| {
				if min_dist > d { (min_dist, min_idx) = (d, i); }
				visited_set.insert(i);
			});
			if self._max_build_frontier_size().is_none() {
				frontier_minheap.clear();
				entry_points.into_iter().for_each(|&mut (d,i)| frontier_minheap.push(d,i));
				let global_ids = if is_layer0 {None} else {Some(&self._global_layer_ids()[layer-1])};
				while let Some((d, v)) = frontier_minheap.pop() {
					if d > min_dist { break; }
					for &(_,j) in graph.view_neighbors(v) {
						if visited_set.insert(j) {
							let j_global = unsafe{(if is_layer0 { j } else { global_ids.unwrap_unchecked()[j.to_usize().unwrap_unchecked()] }).to_usize().unwrap_unchecked()};
							let neighbor_dist = self._get_dist(mat, i, j_global);
							if min_dist > neighbor_dist { (min_dist, min_idx) = (neighbor_dist, j); }
							frontier_minheap.push(neighbor_dist, j);
						}
					}
				}
			} else {
				frontier_dualheap.clear();
				let max_frontier_size = unsafe{self._max_build_frontier_size().unwrap_unchecked()};
				entry_points.into_iter().for_each(|&mut (d,i)| {
					if frontier_dualheap.size() < max_frontier_size {
						frontier_dualheap.push(d,i);
					} else {
						frontier_dualheap.push_pop::<false>(d,i);
					}
				});
				let global_ids = if is_layer0 {None} else {Some(&self._global_layer_ids()[layer-1])};
				while let Some((d, v)) = frontier_dualheap.pop::<true>() {
					if d > min_dist { break; }
					for &(_,j) in graph.view_neighbors(v) {
						if visited_set.insert(j) {
							let j_global = unsafe{(if is_layer0 { j } else { global_ids.unwrap_unchecked()[j.to_usize().unwrap_unchecked()] }).to_usize().unwrap_unchecked()};
							let neighbor_dist = self._get_dist(mat, i, j_global);
							if min_dist > neighbor_dist { (min_dist, min_idx) = (neighbor_dist, j); }
							if frontier_dualheap.size() < max_frontier_size {
								frontier_dualheap.push(neighbor_dist, j);
							} else {
								frontier_dualheap.push_pop::<false>(neighbor_dist, j);
							}
						}
					}
				}
			}
			entry_points.clear();
			entry_points.push((min_dist, min_idx));
		}
	}
	fn finetune_rnn<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, rnn_params: RNNParams) {
		let (lowest_max_degree, higher_max_degree) = self._max_degrees();
		let dist = self._dist().clone();
		let layer_0_params = rnn_params.clone()
		.with_initial_degree(lowest_max_degree)
		.with_reduce_degree(lowest_max_degree);
		let graph = &mut self._mut_graphs()[0];
		let mut rnn = RNNDescentBuilder::from_weighted_graph(
			mat, graph, dist.clone(), layer_0_params,
		);
		rnn.train(mat);
		let (rnn_graph, _) = rnn._into_graph_dist();
		let n_vertices = rnn_graph.n_vertices();
		assert_eq!(n_vertices, graph.n_vertices(), "Graph size mismatch during RNN refinement");
		unsafe {
			(0..n_vertices).for_each(|u| {
				let u = R::from_usize(u).unwrap_unchecked();
				graph.clear_neighbors(u);
				rnn_graph.foreach_neighbor_with_zipped_weight(u, |&w,&v| {
					if u != v {
						graph.add_edge_with_weight(u,v,w);
					}
				});
			});
		}
		let layer_i_params = rnn_params.clone()
		.with_initial_degree(higher_max_degree)
		.with_reduce_degree(higher_max_degree);
		let unsafe_self = self as *mut Self;
		unsafe {
			(*unsafe_self)._mut_graphs().iter_mut().skip(1)
			.zip((*unsafe_self)._global_layer_ids().iter())
			.for_each(|(graph, global_ids)| {
				let copied_mat = mat.get_rows(&global_ids.iter().map(|&i| i.to_usize().unwrap_unchecked()).collect::<Vec<usize>>());
				let mut rnn = RNNDescentBuilder::from_weighted_graph(
					&copied_mat, graph, dist.clone(), layer_i_params,
				);
				rnn.train(&copied_mat);
				let (rnn_graph, _) = rnn._into_graph_dist();
				let n_vertices = rnn_graph.n_vertices();
				assert_eq!(n_vertices, graph.n_vertices(), "Graph size mismatch during RNN refinement");
				(0..n_vertices).for_each(|u| {
					let u = R::from_usize(u).unwrap_unchecked();
					graph.clear_neighbors(u);
					rnn_graph.foreach_neighbor_with_zipped_weight(u, |&w,&v| {
						if u != v {
							graph.add_edge_with_weight(u,v,w);
						}
					});
				});
			});
		}
	}
	fn finetune_sen<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, sen_params: SENParams<F>) {
		let (lowest_max_degree, higher_max_degree) = self._max_degrees();
		let dist = self._dist().clone();
		let layer_0_params = sen_params.clone()
		.with_initial_degree(lowest_max_degree)
		.with_reduce_degree(lowest_max_degree);
		let graph = &mut self._mut_graphs()[0];
		let mut sen = SENDescentBuilder::from_weighted_graph(
			mat, graph, dist.clone(), layer_0_params,
		);
		sen.train(mat);
		let (sen_graph, _) = sen._into_graph_dist();
		let n_vertices = sen_graph.n_vertices();
		assert_eq!(n_vertices, graph.n_vertices(), "Graph size mismatch during SEN refinement");
		unsafe {
			(0..n_vertices).for_each(|u| {
				let u = R::from_usize(u).unwrap_unchecked();
				graph.clear_neighbors(u);
				sen_graph.foreach_neighbor_with_zipped_weight(u, |&w,&v| {
					if u != v {
						graph.add_edge_with_weight(u,v,w);
					}
				});
			});
		}
		let layer_i_params = sen_params.clone()
		.with_initial_degree(higher_max_degree)
		.with_reduce_degree(higher_max_degree);
		let unsafe_self = self as *mut Self;
		unsafe {
			(*unsafe_self)._mut_graphs().iter_mut().skip(1)
			.zip((*unsafe_self)._global_layer_ids().iter())
			.for_each(|(graph, global_ids)| {
				let copied_mat = mat.get_rows(&global_ids.iter().map(|&i| i.to_usize().unwrap_unchecked()).collect::<Vec<usize>>());
				let mut sen = SENDescentBuilder::from_weighted_graph(
					&copied_mat, graph, dist.clone(), layer_i_params,
				);
				sen.train(&copied_mat);
				let (sen_graph, _) = sen._into_graph_dist();
				let n_vertices = sen_graph.n_vertices();
				assert_eq!(n_vertices, graph.n_vertices(), "Graph size mismatch during RNN refinement");
				(0..n_vertices).for_each(|u| {
					let u = R::from_usize(u).unwrap_unchecked();
					graph.clear_neighbors(u);
					sen_graph.foreach_neighbor_with_zipped_weight(u, |&w,&v| {
						if u != v {
							graph.add_edge_with_weight(u,v,w);
						}
					});
				});
			});
		}
	}
}



param_struct!(HNSWParams[Copy, Clone]<F: SyncFloat> {
	higher_max_degree: usize = 50,
	lowest_max_degree: usize = 100,
	max_layers: usize = 10,
	n_parallel_burnin: usize = 0,
	max_build_heap_size: usize = 50,
	max_build_frontier_size: Option<usize> = Some(100),
	level_norm_param_override: Option<f32> = None,
	insert_heuristic: bool = true,
	insert_heuristic_extend: bool = true,
	post_prune_heuristic: bool = false,
	insert_minibatch_size: usize = 100,
	n_rounds: usize = 1,
	finetune_rnn: bool = false,
	finetune_sen: bool = false,
	finetune_rnn_params: RNNParams = RNNParams::new(),
	finetune_sen_params: SENParams<F> = SENParams::new(),
});


pub mod single_threaded {
	use graphidx::graphs::*;
	use graphidx::heaps::*;
	use graphidx::types::*;
	use graphidx::measures::*;
	
	use crate::hnsw::*;

	pub struct HNSWBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
		_phantom: std::marker::PhantomData<F>,
		n_data: usize,
		params: HNSWParams<F>,
		// add_edge_cache: Vec<(R,R,F)>,
		// rem_edge_cache: Vec<(R,R)>,
		n_layers: usize,
		graphs: Vec<HNSWBuildGraph<R,F>>,
		local_layer_ids: Vec<Vec<R>>,
		global_layer_ids: Vec<Vec<R>>,
		dist: Dist,
		level_norm_param: f32,
	}
	impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWBuilder<R, F, Dist> {
		fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
			/* Do nothing for empty datasets */
			if self.n_data == 0 { return; }
			/* Reserve memory in id lookups */
			self.graphs.iter_mut().enumerate().for_each(|(i, g)| {
				let mut expected_size = self.n_data as f32;
				(0..i).for_each(|_| expected_size /= (1.0/self.level_norm_param).exp());
				if i > 0 { expected_size *= 1.2; }
				let expected_size = expected_size.floor() as usize;
				g.reserve(expected_size);
				if i > 0 {
					self.local_layer_ids[i-1].reserve(expected_size);
					self.global_layer_ids[i-1].reserve(expected_size);
				}
			});
			/* Populate all graphs and layers with the first point to avoid empty results anywhere */
			/* This is not in the original paper but who cares */
			(0..self.n_layers).for_each(|i| {
				self.graphs[i].add_node();
				if i > 0 {
					self.local_layer_ids[i-1].push(R::zero());
					self.global_layer_ids[i-1].push(R::zero());
				}
			});
			/* Insert all other points as you would */
			let mut search_hashset = <HashSet<R> as HashSetLike<R>>::new(self.n_data);
			search_hashset.reserve(self.params.max_build_heap_size*2);
			let mut heuristic_hashset = <HashSet<R> as HashSetLike<R>>::new(self.n_data);
			heuristic_hashset.reserve(self.params.lowest_max_degree*self.params.lowest_max_degree);
			let mut search_maxheap = MaxHeap::with_capacity(self.params.max_build_heap_size);
			let mut frontier_minheap = MinHeap::with_capacity(if self.params.max_build_frontier_size.is_some() {0} else {self.params.max_build_heap_size});
			let mut frontier_dualheap = DualHeap::with_capacity(self.params.max_build_frontier_size.unwrap_or(0));
			(1..self.n_data).for_each(|i| {
				self.insert(mat, i, &mut search_hashset, &mut heuristic_hashset, &mut search_maxheap, &mut frontier_minheap, &mut frontier_dualheap);
			});
		}
		fn insert<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, search_hashset: &mut HashSet<R>, heuristic_hashset: &mut HashSet<R>, search_maxheap: &mut MaxHeap<F,R>, frontier_minheap: &mut MinHeap<F,R>, frontier_dualheap: &mut DualHeap<F,R>) {
			let highest_level = random_level(self.level_norm_param, self.n_layers);
			// let top_entry_id = rand::random::<usize>() % self.graphs[self.n_layers-1].n_vertices();
			let top_entry_id = 0; /* The original algorithm uses the first point in the highest layer every time */
			let top_entry_dist = self._get_dist(mat, i, unsafe{self.global_layer_ids[self.n_layers-2][top_entry_id].to_usize().unwrap_unchecked()});
			let mut entry_points = vec![(top_entry_dist, unsafe{R::from_usize(top_entry_id).unwrap_unchecked()})];
			/* Search entry point for the required level */
			(highest_level+1..self.n_layers).rev().for_each(|i_layer| {
				let local_ids = &self.local_layer_ids[i_layer-1];
				self._search_layer(mat, i, i_layer, search_hashset, search_maxheap, frontier_minheap, frontier_dualheap, &mut entry_points, None);
				entry_points.iter_mut().for_each(|(_,j)| *j = local_ids[j.to_usize().unwrap()]);
			});
			(0..highest_level+1).rev().for_each(|i_layer| {
				self._search_layer(mat, i, i_layer, search_hashset, search_maxheap, frontier_minheap, frontier_dualheap, &mut entry_points, Some(self.params.max_build_heap_size));
				self.insert_layer(mat, i, i_layer, &entry_points, heuristic_hashset);
				if i_layer > 0 {
					let local_ids = &self.local_layer_ids[i_layer-1];
					entry_points.iter_mut().for_each(|(_,j)| *j = local_ids[j.to_usize().unwrap()]);
				}
			});
		}
		fn insert_layer<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, layer: usize, neighbors: &Vec<(F,R)>, heuristic_hashset: &mut HashSet<R>) {
			/* Extend layer ID lists */
			if layer > 0 {
				self.local_layer_ids[layer-1].push(unsafe{R::from_usize(self.graphs[layer-1].n_vertices()).unwrap_unchecked()});
				self.global_layer_ids[layer-1].push(unsafe{R::from_usize(i).unwrap_unchecked()});
			};
			/* Translate i into a local (current graph) and a global (dataset/distance computations) ID */
			let (i, i_global) = (unsafe{R::from_usize(self.graphs[layer].n_vertices()).unwrap_unchecked()}, i);
			/* Choose the appropriate max degree */
			let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
			/* Unsafe graph reference to have `self` accessible downstream */
			let graph = &mut self.graphs[layer] as *mut HNSWBuildGraph<R,F>;
			unsafe {
				/* Create a node in the current layer (empty adjacency list) */
				/* This will eventually reach max_neighbors in almost all cases anyways, so just allocate it now */
				(*graph).add_node_with_capacity(max_neighbors);
				let i_adj = (*graph).view_neighbors_vec_mut(i);
				/* Reduce neighbors to set of max_neighbors points by using a heuristic */
				/* Then bidirectionally link i to all of these neighbors */
				/* Afterwards prune neighborhoods of all affected points back down to max_neighbors */
				if !self.params.insert_heuristic {
					i_adj.extend(neighbors.iter().take(max_neighbors).map(|&v| v));
				} else {
					/* Maybe extend candidate set with neighbors of neighbors */
					let mut candidates = neighbors.clone();
					if self.params.insert_heuristic_extend {
						heuristic_hashset.clear();
						candidates.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
						(0..neighbors.len()).for_each(|i_neighbor| {
							(*graph).view_neighbors(R::from_usize(i_neighbor).unwrap_unchecked()).iter().for_each(|&(_,j)| {
								if heuristic_hashset.insert(j) {
									let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
									candidates.push((self._get_dist(mat, i_global, j_global.to_usize().unwrap_unchecked()), j));
								}
							});
						});
						candidates.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
					}
					/* Prune neighborhoods with relative neighbor heuristic */
					let mut tmp_list = Vec::with_capacity(candidates.len());
					i_adj.push(candidates[0]);
					for (dij,j) in candidates.into_iter().skip(1) {
						if i_adj.len() >= max_neighbors { break; }
						if i_adj.iter().all(|&(dik,k)| {
							let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
							let k_global = if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k };
							let djk = self._get_dist(mat, j_global.to_usize().unwrap_unchecked(), k_global.to_usize().unwrap_unchecked());
							dik < djk
						}) {
							i_adj.push((dij,j));
						} else {
							tmp_list.push((dij,j));
						}
					};
					if max_neighbors > i_adj.len() {
						/* Add removed edges back in */
						tmp_list.iter().take(max_neighbors-i_adj.len()).for_each(|&x| i_adj.push(x));
						/* Sort adjacency list to keep up assertion down below */
						i_adj.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
					}
				}
				/* Add backward edges */
				i_adj.iter().for_each(|&(d,j)| {
					let j_adj = (*graph).view_neighbors_vec_mut(j);
					/* Assert that all adjacencies are sorted in ascending order */
					/* Either replace maximum are append */
					let mut curr = if j_adj.len() == max_neighbors {
						j_adj[max_neighbors-1] = (d,i);
						max_neighbors-1
					} else {
						j_adj.push((d,i));
						j_adj.len()-1
					};
					/* Insertion sort the new entry to the correct position */
					while curr > 0 && (*j_adj)[curr].0 < (*j_adj)[curr-1].0 {
						j_adj.swap(curr, curr-1);
						curr -= 1;
					}
				});
			}
		}
	}
	impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for HNSWBuilder<R, F, Dist> {
		type Params = HNSWParams<F>;
		type Graph = HNSWBuildGraph<R,F>;
		#[inline(always)]
		fn _mut_graphs(&mut self) -> &mut Vec<HNSWBuildGraph<R,F>> { &mut self.graphs }
		#[inline(always)]
		fn _graphs(&self) -> &Vec<HNSWBuildGraph<R,F>> { &self.graphs }
		#[inline(always)]
		fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
		#[inline(always)]
		fn _dist(&self) -> &Dist { &self.dist }
		#[inline(always)]
		fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
		#[inline(always)]
		fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
		#[inline(always)]
		fn _into_parts(self) -> (Vec<HNSWBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
			(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
		}
		#[inline(always)]
		fn _max_degrees(&self) -> (usize, usize) {
			(self.params.lowest_max_degree, self.params.higher_max_degree)
		}
		fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
			let n_data = mat.n_rows();
			assert!(n_data < R::max_value().to_usize().unwrap());
			let level_norm_param = params.level_norm_param_override.unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
			/* By the law of large numbers, this is the appropriate number of layers */
			let n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1).min(params.max_layers);
			let mut builder = Self {
				_phantom: std::marker::PhantomData,
				n_data,
				params,
				// add_edge_cache: Vec::new(),
				// rem_edge_cache: Vec::new(),
				n_layers,
				graphs: (0..n_layers).map(|_| HNSWBuildGraph::new()).collect(),
				local_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
				global_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
				dist,
				level_norm_param,
			};
			builder.train(mat);
			builder
		}
	}



	pub struct HNSWHeapBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
		_phantom: std::marker::PhantomData<F>,
		n_data: usize,
		params: HNSWParams<F>,
		// add_edge_cache: Vec<(R,R,F)>,
		// rem_edge_cache: Vec<(R,R)>,
		n_layers: usize,
		graphs: Vec<HNSWHeapBuildGraph<R,F>>,
		local_layer_ids: Vec<Vec<R>>,
		global_layer_ids: Vec<Vec<R>>,
		dist: Dist,
		level_norm_param: f32,
	}
	impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWHeapBuilder<R, F, Dist> {
		fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
			/* Do nothing for empty datasets */
			if self.n_data == 0 { return; }
			/* Reserve memory in id lookups */
			self.graphs.iter_mut().enumerate().for_each(|(i, g)| {
				let mut expected_size = self.n_data as f32;
				(0..i).for_each(|_| expected_size /= (1.0/self.level_norm_param).exp());
				if i > 0 { expected_size *= 1.2; }
				let expected_size = expected_size.floor() as usize;
				g.reserve(expected_size);
				if i > 0 {
					self.local_layer_ids[i-1].reserve(expected_size);
					self.global_layer_ids[i-1].reserve(expected_size);
				}
			});
			/* Populate all graphs and layers with the first point to avoid empty results anywhere */
			/* This is not in the original paper but who cares */
			(0..self.n_layers).for_each(|i| {
				self.graphs[i].add_node();
				if i > 0 {
					self.local_layer_ids[i-1].push(R::zero());
					self.global_layer_ids[i-1].push(R::zero());
				}
			});
			/* Insert all other points as you would */
			let mut search_hashset = <HashSet<R> as HashSetLike<R>>::new(self.n_data);
			search_hashset.reserve(self.params.max_build_heap_size*2);
			let mut heuristic_hashset = <HashSet<R> as HashSetLike<R>>::new(self.n_data);
			heuristic_hashset.reserve(self.params.lowest_max_degree*self.params.lowest_max_degree);
			let mut search_maxheap = MaxHeap::with_capacity(self.params.max_build_heap_size);
			let mut frontier_minheap = MinHeap::with_capacity(if self.params.max_build_frontier_size.is_some() {0} else {self.params.max_build_heap_size});
			let mut frontier_dualheap = DualHeap::with_capacity(self.params.max_build_frontier_size.unwrap_or(0));
			(1..self.n_data).for_each(|i| {
				self.insert(mat, i, &mut search_hashset, &mut heuristic_hashset, &mut search_maxheap, &mut frontier_minheap, &mut frontier_dualheap);
			});
		}
		fn insert<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, search_hashset: &mut HashSet<R>, heuristic_hashset: &mut HashSet<R>, search_maxheap: &mut MaxHeap<F,R>, frontier_minheap: &mut MinHeap<F,R>, frontier_dualheap: &mut DualHeap<F,R>) {
			let highest_level = random_level(self.level_norm_param, self.n_layers);
			// let top_entry_id = rand::random::<usize>() % self.graphs[self.n_layers-1].n_vertices();
			let top_entry_id = 0; /* The original algorithm uses the first point in the highest layer every time */
			let top_entry_dist = self._get_dist(mat, i, unsafe{self.global_layer_ids[self.n_layers-2][top_entry_id].to_usize().unwrap_unchecked()});
			let mut entry_points = vec![(top_entry_dist, unsafe{R::from_usize(top_entry_id).unwrap_unchecked()})];
			/* Search entry point for the required level */
			(highest_level+1..self.n_layers).rev().for_each(|i_layer| {
				let local_ids = &self.local_layer_ids[i_layer-1];
				self._search_layer(mat, i, i_layer, search_hashset, search_maxheap, frontier_minheap, frontier_dualheap, &mut entry_points, None);
				entry_points.iter_mut().for_each(|(_,j)| *j = local_ids[j.to_usize().unwrap()]);
			});
			(0..highest_level+1).rev().for_each(|i_layer| {
				self._search_layer(mat, i, i_layer, search_hashset, search_maxheap, frontier_minheap, frontier_dualheap, &mut entry_points, Some(self.params.max_build_heap_size));
				self.insert_layer(mat, i, i_layer, &entry_points, heuristic_hashset);
				if i_layer > 0 {
					let local_ids = &self.local_layer_ids[i_layer-1];
					entry_points.iter_mut().for_each(|(_,j)| *j = local_ids[j.to_usize().unwrap()]);
				}
			});
		}
		fn insert_layer<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, layer: usize, neighbors: &Vec<(F,R)>, heuristic_hashset: &mut HashSet<R>) {
			/* Extend layer ID lists */
			if layer > 0 {
				self.local_layer_ids[layer-1].push(unsafe{R::from_usize(self.graphs[layer-1].n_vertices()).unwrap_unchecked()});
				self.global_layer_ids[layer-1].push(unsafe{R::from_usize(i).unwrap_unchecked()});
			};
			/* Translate i into a local (current graph) and a global (dataset/distance computations) ID */
			let (i, i_global) = (unsafe{R::from_usize(self.graphs[layer].n_vertices()).unwrap_unchecked()}, i);
			/* Choose the appropriate max degree */
			let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
			/* Unsafe graph reference to have `self` accessible downstream */
			let graph = &mut self.graphs[layer] as *mut HNSWHeapBuildGraph<R,F>;
			unsafe {
				/* Create a node in the current layer (empty adjacency list) */
				/* This will eventually reach max_neighbors in almost all cases anyways, so just allocate it now */
				(*graph).add_node_with_capacity(max_neighbors);
				let i_adj = (*graph).view_neighbors_heap_mut(i);
				/* Reduce neighbors to set of max_neighbors points by using a heuristic */
				/* Then bidirectionally link i to all of these neighbors */
				/* Afterwards prune neighborhoods of all affected points back down to max_neighbors */
				if !self.params.insert_heuristic {
					neighbors.iter().take(max_neighbors).for_each(|&(d,j)| i_adj.push(d,j));
				} else {
					/* Maybe extend candidate set with neighbors of neighbors */
					let mut candidates = neighbors.clone();
					if self.params.insert_heuristic_extend {
						heuristic_hashset.clear();
						candidates.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
						(0..neighbors.len()).for_each(|i_neighbor| {
							(*graph).view_neighbors(R::from_usize(i_neighbor).unwrap_unchecked()).iter().for_each(|&(_,j)| {
								if heuristic_hashset.insert(j) {
									let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
									candidates.push((self._get_dist(mat, i_global, j_global.to_usize().unwrap_unchecked()), j));
								}
							});
						});
						candidates.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
					}
					/* Prune neighborhoods with relative neighbor heuristic */
					let mut tmp_list = Vec::with_capacity(candidates.len());
					i_adj.push(candidates[0].0,candidates[0].1);
					for (dij,j) in candidates.into_iter().skip(1) {
						if i_adj.size() >= max_neighbors { break; }
						if i_adj.iter().all(|&(dik,k)| {
							let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
							let k_global = if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k };
							let djk = self._get_dist(mat, j_global.to_usize().unwrap_unchecked(), k_global.to_usize().unwrap_unchecked());
							dik < djk
						}) {
							i_adj.push(dij,j);
						} else {
							tmp_list.push((dij,j));
						}
					};
					if max_neighbors > i_adj.size() {
						/* Add removed edges back in */
						tmp_list.iter().take(max_neighbors-i_adj.size()).for_each(|&x| i_adj.push(x.0,x.1));
					}
				}
				/* Add backward edges */
				i_adj.iter().for_each(|&(d,j)| {
					let j_adj = (*graph).view_neighbors_heap_mut(j);
					/* Either replace maximum are append */
					if j_adj.size() < max_neighbors {
						j_adj.push(d,i);
					} else {
						j_adj.push_pop(d,i);
					}
				});
			}
		}
	}
	impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for HNSWHeapBuilder<R, F, Dist> {
		type Params = HNSWParams<F>;
		type Graph = HNSWHeapBuildGraph<R,F>;
		#[inline(always)]
		fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
		#[inline(always)]
		fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
		#[inline(always)]
		fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
		#[inline(always)]
		fn _dist(&self) -> &Dist { &self.dist }
		#[inline(always)]
		fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
		#[inline(always)]
		fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
		#[inline(always)]
		fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
			(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
		}
		#[inline(always)]
		fn _max_degrees(&self) -> (usize, usize) {
			(self.params.lowest_max_degree, self.params.higher_max_degree)
		}
		fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
			let n_data = mat.n_rows();
			assert!(n_data < R::max_value().to_usize().unwrap());
			let level_norm_param = params.level_norm_param_override.unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
			/* By the law of large numbers, this is the appropriate number of layers */
			let n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1).min(params.max_layers);
			let mut builder = Self {
				_phantom: std::marker::PhantomData,
				n_data,
				params,
				// add_edge_cache: Vec::new(),
				// rem_edge_cache: Vec::new(),
				n_layers,
				graphs: (0..n_layers).map(|_| HNSWHeapBuildGraph::new()).collect(),
				local_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
				global_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
				dist,
				level_norm_param,
			};
			builder.train(mat);
			builder
		}

	}



	pub struct HNSWHeapBuilder2<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
		_phantom: std::marker::PhantomData<F>,
		n_data: usize,
		params: HNSWParams<F>,
		// add_edge_cache: Vec<(R,R,F)>,
		// rem_edge_cache: Vec<(R,R)>,
		n_layers: usize,
		graphs: Vec<HNSWHeapBuildGraph<R,F>>,
		local_layer_ids: Vec<Vec<R>>,
		global_layer_ids: Vec<Vec<R>>,
		dist: Dist,
		level_norm_param: f32,
	}
	impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWHeapBuilder2<R, F, Dist> {
		fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
			/* Do nothing for empty datasets */
			if self.n_data == 0 { return; }
			/* Precompute all point levels */
			let mut layer_sizes = vec![0; self.n_layers];
			let mut point_levels = Vec::with_capacity(self.n_data);
			unsafe{point_levels.set_len(self.n_data)};
			let mut point_level_pos = Vec::with_capacity(self.n_data);
			unsafe{point_level_pos.set_len(self.n_data)};
			/* Always insert first point on highest level */
			point_levels[0] = self.n_layers-1;
			layer_sizes[self.n_layers-1] = 1;
			/* Insert all other points */
			(1..self.n_data).for_each(|i| {
				let level = random_level(self.level_norm_param, self.n_layers);
				point_levels[i] = level;
				layer_sizes[level] += 1;
			});
			/* Aggregate layer sizes such that the bottom layer has size N */
			(0..self.n_layers-1).rev().for_each(|i| layer_sizes[i] += layer_sizes[i+1]);
			debug_assert!(layer_sizes[0] == self.n_data);
			/* Initialize lowest level */
			let graph0 = &mut self.graphs[0];
			graph0.reserve(self.n_data);
			(0..self.n_data).for_each(|_| graph0.add_node_with_capacity(self.params.lowest_max_degree));
			/* Initialize all other levels */
			let mut layer_offsets = vec![0usize; self.n_layers];
			unsafe {
				(1..self.n_layers).for_each(|i_layer| {
					let layer_size = layer_sizes[i_layer];
					let local_ids = self.local_layer_ids.get_unchecked_mut(i_layer-1);
					local_ids.reserve(layer_size);
					local_ids.set_len(layer_size);
					let global_ids = self.global_layer_ids.get_unchecked_mut(i_layer-1);
					global_ids.reserve(layer_size);
					global_ids.set_len(layer_size);
					let graph = &mut self.graphs[i_layer];
					graph.reserve(layer_size);
					(0..layer_size).for_each(|_| graph.add_node_with_capacity(self.params.higher_max_degree));
				});
				point_levels.iter().enumerate().for_each(|(i, &level)| {
					point_level_pos[i] = layer_offsets[level];
					(0..level+1).rev().for_each(|i_layer| {
						if i_layer > 0 {
							self.local_layer_ids.get_unchecked_mut(i_layer-1)[layer_offsets[i_layer]] = R::from_usize(layer_offsets[i_layer-1]).unwrap_unchecked();
							self.global_layer_ids.get_unchecked_mut(i_layer-1)[layer_offsets[i_layer]] = R::from_usize(i).unwrap_unchecked();
						}
						layer_offsets[i_layer] += 1;
					});
				});
				#[cfg(debug_assertions)]
				{
					layer_offsets.iter().zip(layer_sizes.iter()).for_each(|(a, b)| assert_eq!(a, b));
					point_levels.iter().zip(point_level_pos.iter()).enumerate().for_each(|(i,(&level, &pos))| {
						assert!(layer_sizes[level] > pos);
						let mut pos = pos;
						(1..level+1).rev().for_each(|i_layer| pos = self.local_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked());
						assert_eq!(i, pos);
					});
				}
			}
			/* Insert all other points as you would */
			let mut search_hashset = <HashSet<R> as HashSetLike<R>>::new(self.n_data);
			search_hashset.reserve(self.params.max_build_heap_size*2);
			let mut heuristic_hashset = <HashSet<R> as HashSetLike<R>>::new(self.n_data);
			heuristic_hashset.reserve(self.params.lowest_max_degree*self.params.lowest_max_degree);
			let mut search_maxheap = MaxHeap::with_capacity(self.params.max_build_heap_size);
			let mut frontier_minheap = MinHeap::with_capacity(if self.params.max_build_frontier_size.is_some() {0} else {self.params.max_build_heap_size});
			let mut frontier_dualheap = DualHeap::with_capacity(self.params.max_build_frontier_size.unwrap_or(0));
			let mut entry_points = Vec::with_capacity(self.params.max_build_heap_size);
			(1..self.n_data).for_each(|i| {
				self.insert(mat, point_level_pos[i], i, point_levels[i], &mut entry_points, &mut search_hashset, &mut heuristic_hashset, &mut search_maxheap, &mut frontier_minheap, &mut frontier_dualheap);
			});
		}
		fn insert<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, mut i: usize, i_global: usize, level: usize, entry_points: &mut Vec<(F,R)>, search_hashset: &mut HashSet<R>, heuristic_hashset: &mut HashSet<R>, search_maxheap: &mut MaxHeap<F,R>, frontier_minheap: &mut MinHeap<F,R>, frontier_dualheap: &mut DualHeap<F,R>) {
			let highest_level = level;
			let top_entry_id = 0; /* The original algorithm uses the first point in the highest layer every time */
			let top_entry_dist = self._get_dist(mat, i_global, unsafe{self.global_layer_ids[self.n_layers-2][top_entry_id].to_usize().unwrap_unchecked()});
			entry_points.clear();
			entry_points.push((top_entry_dist, unsafe{R::from_usize(top_entry_id).unwrap_unchecked()}));
			/* Search entry point for the required level */
			(highest_level+1..self.n_layers).rev().for_each(|i_layer| {
				let local_ids = &self.local_layer_ids[i_layer-1];
				self._search_layer(mat, i_global, i_layer, search_hashset, search_maxheap, frontier_minheap, frontier_dualheap, entry_points, None);
				entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
			});
			(0..highest_level+1).rev().for_each(|i_layer| {
				self._search_layer(mat, i_global, i_layer, search_hashset, search_maxheap, frontier_minheap, frontier_dualheap, entry_points, Some(self.params.max_build_heap_size));
				self.insert_layer(mat, i, i_global, i_layer, &entry_points, heuristic_hashset);
				if i_layer > 0 {
					let local_ids = &self.local_layer_ids[i_layer-1];
					entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
					i = unsafe{local_ids[i].to_usize().unwrap_unchecked()};
				}
			});
		}
		fn insert_layer<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, i_global: usize, layer: usize, neighbors: &Vec<(F,R)>, heuristic_hashset: &mut HashSet<R>) {
			/* Translate i into a local (current graph) and a global (dataset/distance computations) ID */
			let i = unsafe{R::from_usize(i).unwrap_unchecked()};
			/* Choose the appropriate max degree */
			let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
			/* Unsafe graph reference to have `self` accessible downstream */
			let graph = &mut self.graphs[layer] as *mut HNSWHeapBuildGraph<R,F>;
			unsafe {
				let i_adj = (*graph).view_neighbors_heap_mut(i);
				/* Reduce neighbors to set of max_neighbors points by using a heuristic */
				/* Then bidirectionally link i to all of these neighbors */
				/* Afterwards prune neighborhoods of all affected points back down to max_neighbors */
				if !self.params.insert_heuristic {
					neighbors.iter().take(max_neighbors).for_each(|&(d,j)| i_adj.push(d,j));
				} else {
					/* Maybe extend candidate set with neighbors of neighbors */
					let mut candidates = neighbors.clone();
					if self.params.insert_heuristic_extend {
						heuristic_hashset.clear();
						candidates.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
						(0..neighbors.len()).for_each(|i_neighbor| {
							(*graph).view_neighbors(R::from_usize(i_neighbor).unwrap_unchecked()).iter().for_each(|&(_,j)| {
								if heuristic_hashset.insert(j) {
									let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
									candidates.push((self._get_dist(mat, i_global, j_global.to_usize().unwrap_unchecked()), j));
								}
							});
						});
						candidates.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
					}
					/* Prune neighborhoods with relative neighbor heuristic */
					let mut tmp_list = Vec::with_capacity(candidates.len());
					i_adj.push(candidates[0].0,candidates[0].1);
					for (dij,j) in candidates.into_iter().skip(1) {
						if i_adj.size() >= max_neighbors { break; }
						if i_adj.iter().all(|&(dik,k)| {
							let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
							let k_global = if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k };
							let djk = self._get_dist(mat, j_global.to_usize().unwrap_unchecked(), k_global.to_usize().unwrap_unchecked());
							dik < djk
						}) {
							i_adj.push(dij,j);
						} else {
							tmp_list.push((dij,j));
						}
					};
					if max_neighbors > i_adj.size() {
						/* Add removed edges back in */
						tmp_list.iter().take(max_neighbors-i_adj.size()).for_each(|&x| i_adj.push(x.0,x.1));
					}
				}
				/* Add backward edges */
				i_adj.iter().for_each(|&(d,j)| {
					let j_adj = (*graph).view_neighbors_heap_mut(j);
					/* Either replace maximum are append */
					if j_adj.size() < max_neighbors {
						j_adj.push(d,i);
					} else {
						j_adj.push_pop(d,i);
					}
				});
			}
		}
	}
	impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for HNSWHeapBuilder2<R, F, Dist> {
		type Params = HNSWParams<F>;
		type Graph = HNSWHeapBuildGraph<R,F>;
		#[inline(always)]
		fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
		#[inline(always)]
		fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
		#[inline(always)]
		fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
		#[inline(always)]
		fn _dist(&self) -> &Dist { &self.dist }
		#[inline(always)]
		fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
		#[inline(always)]
		fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
		#[inline(always)]
		fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
			(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
		}
		#[inline(always)]
		fn _max_degrees(&self) -> (usize, usize) {
			(self.params.lowest_max_degree, self.params.higher_max_degree)
		}
		fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
			let n_data = mat.n_rows();
			assert!(n_data < R::max_value().to_usize().unwrap());
			let level_norm_param = params.level_norm_param_override.unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
			/* By the law of large numbers, this is the appropriate number of layers */
			let n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1).min(params.max_layers);
			let mut builder = Self {
				_phantom: std::marker::PhantomData,
				n_data,
				params,
				// add_edge_cache: Vec::new(),
				// rem_edge_cache: Vec::new(),
				n_layers,
				graphs: (0..n_layers).map(|_| HNSWHeapBuildGraph::new()).collect(),
				local_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
				global_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
				dist,
				level_norm_param,
			};
			builder.train(mat);
			builder
		}

	}
}



macro_rules! with_guard {
	($self:ident, $num:ident, $body:expr) => {{
		let guard = $self.node_locks[$num].lock().unwrap();
		let ret = $body;
		drop(guard);
		ret
	}};
}
#[allow(unused)]
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


struct HNSWThreadCache<R: SyncUnsignedInteger, F: SyncFloat> {
	entry_points: Vec<(F,R)>,
	search_hashsets: Vec<HashOrBitset<R>>,
	heuristic_hashsets: Vec<HashOrBitset<R>>,
	search_maxheap: MaxHeap<F,R>,
	frontier_minheap: MinHeap<F,R>,
	frontier_dualheap: DualHeap<F,R>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat> HNSWThreadCache<R,F> {
	fn new(layer_sizes: &Vec<usize>, max_build_heap_size: usize, _lowest_max_degree: usize, max_build_frontier_size: Option<usize>) -> Self {
		let entry_points = Vec::with_capacity(max_build_heap_size);
		let search_hashsets = layer_sizes.iter().map(|&n| HashOrBitset::new(n)).collect();
		let heuristic_hashsets = layer_sizes.iter().map(|&n| HashOrBitset::new(n)).collect();
		let search_maxheap = MaxHeap::with_capacity(max_build_heap_size);
		let frontier_minheap = MinHeap::with_capacity(if max_build_frontier_size.is_some() {0} else {max_build_heap_size});
		let frontier_dualheap = DualHeap::with_capacity(max_build_frontier_size.unwrap_or(0));
		HNSWThreadCache {
			entry_points,
			search_hashsets,
			heuristic_hashsets,
			search_maxheap,
			frontier_minheap,
			frontier_dualheap,
		}
	}
}
pub struct HNSWParallelHeapBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	n_threads: usize,
	params: HNSWParams<F>,
	node_locks: Vec<Mutex<()>>,
	n_layers: usize,
	graphs: Vec<HNSWHeapBuildGraph<R,F>>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	dist: Dist,
	level_norm_param: f32,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWParallelHeapBuilder<R, F, Dist> {
	fn initialize_structure(&mut self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
		/* Precompute all point levels */
		let layer_sizes = (0..self.n_layers).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
		let mut point_levels = Vec::with_capacity(self.n_data);
		unsafe{point_levels.set_len(self.n_data)};
		let mut point_level_pos: Vec<usize> = Vec::with_capacity(self.n_data);
		unsafe{point_level_pos.set_len(self.n_data)};
		/* Always insert first point on highest level */
		point_levels[0] = self.n_layers-1;
		layer_sizes[self.n_layers-1].store(1, Relaxed);
		/* Insert all other points */
		let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
		point_levels[1..].chunks_mut(chunk_size).par_bridge().for_each(|point_level_chunk| {
			point_level_chunk.iter_mut().for_each(|point_level_ref| {
				let level = random_level(self.level_norm_param, self.n_layers);
				*point_level_ref = level;
				layer_sizes[level].fetch_add(1, Relaxed);
			})
		});
		/* Remove atomic wrappers */
		let mut layer_sizes = layer_sizes.into_iter().map(|x| x.into_inner()).collect::<Vec<_>>();
		/* Aggregate layer sizes such that the bottom layer has size N */
		(0..self.n_layers-1).rev().for_each(|i| layer_sizes[i] += layer_sizes[i+1]);
		debug_assert!(layer_sizes[0] == self.n_data);
		/* Initialize lowest level */
		let graph0 = &mut self.graphs[0];
		graph0.reserve(self.n_data);
		(0..self.n_data).for_each(|_| graph0.add_node_with_capacity(self.params.lowest_max_degree));
		/* Initialize all other levels */
		unsafe {
			/* Preallocate graphs and ID lookup memory (warning: uninitialized RAM, handle with care!) */
			layer_sizes.iter().skip(1)
			.zip(self.local_layer_ids.iter_mut())
			.zip(self.global_layer_ids.iter_mut())
			.zip(self.graphs.iter_mut().skip(1))
			.par_bridge()
			.for_each(|(((&layer_size, local_ids), global_ids), graph)| {
				local_ids.reserve(layer_size);
				local_ids.set_len(layer_size);
				global_ids.reserve(layer_size);
				global_ids.set_len(layer_size);
				graph.reserve(layer_size);
				(0..layer_size).for_each(|_| graph.add_node_with_capacity(self.params.higher_max_degree));
			});
			/* Set level pos for layer 0 */
			point_level_pos.chunks_mut(chunk_size)
			.zip(point_levels.chunks(chunk_size))
			.enumerate().par_bridge().for_each(|(i_chunk, (point_level_pos_chunk, point_level_chunk))| {
				point_level_pos_chunk.iter_mut().zip(point_level_chunk.iter()).enumerate()
				.filter(|(_,(_,&level))| level==0).for_each(|(i, (point_level_pos_ref, _))| {
					*point_level_pos_ref = i_chunk*chunk_size + i;
				});
			});
			/* Set level pos for all higher levels */
			self.local_layer_ids.iter_mut()
			.zip(self.global_layer_ids.iter_mut())
			.enumerate().par_bridge().for_each(|(i_layer,(local_ids, global_ids))| {
				/* Unsafe reference for (unsafe) parallel write access */
				let unsafe_point_level_pos = std::ptr::addr_of!(point_level_pos) as *mut Vec<usize>;
				/* Incrementing enumerator as the ID arrays start at layer 1 */
				let i_layer = i_layer+1;
				/* Counting local IDs for current and next layer to properly connect them in ID lookups */
				let next_level = i_layer-1;
				let curr_level = i_layer;
				let mut next_layer_offset = R::zero();
				let mut curr_layer_offset = 0;
				/* Setting up local and global ID lookups */
				point_levels.iter().enumerate().for_each(|(i, &level)| {
					if level >= next_level {
						/* Point i requires an entry in the lookups of the current layer */
						if level >= curr_level {
							local_ids[curr_layer_offset] = next_layer_offset;
							global_ids[curr_layer_offset] = R::from_usize(i).unwrap_unchecked();
							if level == curr_level { (&mut *unsafe_point_level_pos)[i] = curr_layer_offset; }
							curr_layer_offset += 1;
						}
						/* Point i requires an entry in the lookups of the next layer */
						next_layer_offset += R::one();
					}
				});
				/* Ensure that all IDs between 0 and N were used for this and next layer */
				debug_assert!(curr_layer_offset == layer_sizes[curr_level]);
				debug_assert!(next_layer_offset.to_usize().unwrap() == layer_sizes[next_level]);
			});
			#[cfg(debug_assertions)]
			{
				point_levels.iter().zip(point_level_pos.iter()).enumerate().for_each(|(i,(&level, &pos))| {
					assert!(layer_sizes[level] > pos);
					let mut pos = pos;
					(1..level+1).rev().for_each(|i_layer| {
						assert_eq!(i, self.global_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked());
						pos = self.local_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked();
					});
					assert_eq!(i, pos);
				});
			}
		}
		(layer_sizes, point_levels, point_level_pos)
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		/* Do nothing for empty datasets */
		if self.n_data == 0 { return; }
		/* Preallocate graphs and ID lookups and assign levels and level positions for each input */
		let (layer_sizes, point_levels, point_level_pos) = self.initialize_structure();
		/* Burnin single thread for a few samples */
		let n_burnin: usize = self.params.n_parallel_burnin.min(self.n_data-1);
		let mut burnin_cache = HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size);
		(1..n_burnin+1).for_each(|i| {
			self.insert(0, mat, point_level_pos[i], i, point_levels[i], &mut burnin_cache);
		});
		/* Insert edges to the graphs as per HNSW insertion rules */
		let mut thread_caches = (0..self.n_threads).map(|_| HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size)).collect::<Vec<_>>();
		let chunk_size = (self.n_data-n_burnin + self.n_threads - 1) / self.n_threads;
		(1+n_burnin..self.n_data).step_by(chunk_size)
		.map(|start| start..(start+chunk_size).min(self.n_data))
		.zip(thread_caches.iter_mut())
		.par_bridge().for_each(|(chunk, thread_cache)| {
			let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
			#[cfg(debug_assertions)]
			unsafe { /* Ensure that the self ref actually works */
				assert_eq!(self.n_data, (*unsafe_self_ref).n_data);
				assert_eq!(self.n_layers, (*unsafe_self_ref).n_layers);
				assert_eq!(self.params.max_build_heap_size, (*unsafe_self_ref).params.max_build_heap_size);
				assert_eq!(self.params.lowest_max_degree, (*unsafe_self_ref).params.lowest_max_degree);
				assert_eq!(self.params.max_build_frontier_size, (*unsafe_self_ref).params.max_build_frontier_size);
				assert_eq!(self.params.insert_heuristic, (*unsafe_self_ref).params.insert_heuristic);
				assert_eq!(self.params.insert_heuristic_extend, (*unsafe_self_ref).params.insert_heuristic_extend);
			}
			unsafe{
				chunk.for_each(|i| {
					(*unsafe_self_ref).insert(0, mat, point_level_pos[i], i, point_levels[i], thread_cache);
				});
			}
		});
		(1..self.params.n_rounds).for_each(|i_round| {
			let perm_gen = RandomPermutationGenerator::new(self.n_data-1, 4);
			let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
			(0..self.n_data-1).step_by(chunk_size)
			.map(|start| start..(start+chunk_size).min(self.n_data-1))
			.zip(thread_caches.iter_mut())
			.par_bridge().for_each(|(chunk, thread_cache)| {
				let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
				unsafe{
					chunk.for_each(|i| {
						let i = perm_gen.apply_rounds(i)+1;
						(*unsafe_self_ref).insert(i_round, mat, point_level_pos[i], i, point_levels[i], thread_cache);
					});
				}
			});
		});
		if self.params.finetune_rnn {
			self.finetune_rnn(mat, self.params.finetune_rnn_params.clone());
		}
		if self.params.finetune_sen {
			self.finetune_sen(mat, self.params.finetune_sen_params.clone());
		}
		if self.params.post_prune_heuristic {
			self.heuristic_post_prune(mat);
		}
	}
	fn insert<M: MatrixDataSource<F>+Sync>(&mut self, _i_round: usize, mat: &M, mut i: usize, i_global: usize, level: usize, thread_cache: &mut HNSWThreadCache<R,F>) {
		/* We just assume that the global index 0 is the root entry point */
		let top_entry_dist = self._get_dist(mat, i_global, 0);
		thread_cache.entry_points.clear();
		thread_cache.entry_points.push((top_entry_dist, R::zero()));
		/* Search entry point for the required level */
		(level+1..self.n_layers).rev().for_each(|i_layer| {
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, None);
			let local_ids = &self.local_layer_ids[i_layer-1];
			thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
		});
		(0..level+1).rev().for_each(|i_layer| {
			// if _i_round > 0 { self.graphs[i_layer].view_neighbors_heap_mut(unsafe{R::from(i).unwrap_unchecked()}).clear(); }
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, Some(self.params.max_build_heap_size));
			self.insert_layer(mat, i, i_global, i_layer, thread_cache);
			if i_layer > 0 {
				let local_ids = &self.local_layer_ids[i_layer-1];
				thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
				i = unsafe{local_ids[i].to_usize().unwrap_unchecked()};
			}
		});
	}
	fn insert_layer<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, i_global: usize, layer: usize, thread_cache: &mut HNSWThreadCache<R,F>) {
		let neighbors = &mut thread_cache.entry_points;
		debug_assert_eq!(neighbors.len(), neighbors.iter().map(|(_,j)| j).collect::<std::collections::HashSet<_>>().len());
		let heuristic_hashset = unsafe{thread_cache.heuristic_hashsets.get_unchecked_mut(layer)};
		/* Translate i into a local (current graph) and a global (dataset/distance computations) ID */
		let i = unsafe{R::from_usize(i).unwrap_unchecked()};
		/* Choose the appropriate max degree */
		let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
		/* Unsafe graph reference to have `self` accessible downstream */
		let graph = &mut self.graphs[layer] as *mut HNSWHeapBuildGraph<R,F>;
		unsafe {
			let i_adj = (*graph).view_neighbors_heap_mut(i);
			/* Reduce neighbors to set of max_neighbors points by using a heuristic */
			/* Then bidirectionally link i to all of these neighbors */
			/* Afterwards prune neighborhoods of all affected points back down to max_neighbors */
			if !self.params.insert_heuristic {
				heuristic_hashset.clear();
				heuristic_hashset.insert(i);
				with_guard!(self, i_global, {
					i_adj.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					neighbors.iter()
					/* This should almost never matter, but this node can be found by another
					 * searching thread while creating the adjacency list. */
					.filter(|&&(_,j)| heuristic_hashset.insert(j))
					.take(max_neighbors)
					.for_each(|&(d,j)| {
						if i_adj.size() < max_neighbors {
							i_adj.push(d,j);
						} else {
							i_adj.push_pop(d,j);
						}
					});
				});
			} else {
				/* Maybe extend candidate set with neighbors of neighbors */
				let mut candidates = neighbors.clone();
				/* Add previous neighbors to the candidate set */
				candidates.extend(i_adj.iter());
				i_adj.clear();
				if self.params.insert_heuristic_extend {
					heuristic_hashset.clear();
					candidates.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					(0..neighbors.len()).for_each(|i_neighbor| {
						(*graph).view_neighbors(R::from_usize(i_neighbor).unwrap_unchecked()).iter().for_each(|&(_,j)| {
							if heuristic_hashset.insert(j) {
								let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
								candidates.push((self._get_dist(mat, i_global, j_global.to_usize().unwrap_unchecked()), j));
							}
						});
					});
				}
				/* Make sure that the candidates are truly sorted and unique */
				candidates.sort_unstable_by(|a,b| {
					let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
					if dist_cmp.is_eq() { a.1.cmp(&b.1) } else { dist_cmp }
				});
				remove_duplicates_with_key(&mut candidates, |(_,j)| j);
				/* Prune neighborhoods with relative neighbor heuristic */
				heuristic_hashset.clear();
				heuristic_hashset.insert(i);
				let mut tmp_list = Vec::with_capacity(candidates.len());
				let mut cand_iter = candidates.into_iter().peekable();
				with_guard!(self, i_global, {
					while cand_iter.peek().is_some() && heuristic_hashset.contains(&cand_iter.peek().unwrap_unchecked().1) { cand_iter.next(); }
					if cand_iter.peek().is_some() {
						let (dij0,j0) = cand_iter.next().unwrap_unchecked();
						i_adj.push(dij0,j0);
						heuristic_hashset.insert(j0);
						for (dij,j) in cand_iter {
							if i_adj.size() >= max_neighbors { break; }
							if !heuristic_hashset.insert(j) { continue; }
							if i_adj.iter().all(|&(dik,k)| {
								let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
								let k_global = if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k };
								let djk = self._get_dist(mat, j_global.to_usize().unwrap_unchecked(), k_global.to_usize().unwrap_unchecked());
								dik < djk
								// tri_to_cos(dij.to_f32().unwrap_unchecked(),dik.to_f32().unwrap_unchecked(),djk.to_f32().unwrap_unchecked(),true) < 0.5
							}) {
								i_adj.push(dij,j);
							} else {
								tmp_list.push((dij,j));
							}
						};
						if max_neighbors > i_adj.size() {
							/* Add removed edges back in */
							tmp_list.iter().take(max_neighbors-i_adj.size()).for_each(|&x| i_adj.push(x.0,x.1));
						}
					}
				});
			}
			/* Add backward edges */
			i_adj.iter().for_each(|&(d,j)| {
				let j_adj = (*graph).view_neighbors_heap_mut(j);
				let j_global = (if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j }).to_usize().unwrap_unchecked();
				/* Either replace maximum are append */
				with_guard!(self, j_global, {
					/* Due to race conditions we must ensure, that i is not in this adjacency list */
					if j_adj.iter().all(|&(_,k)| k != i) {
						if j_adj.size() < max_neighbors {
							j_adj.push(d,i);
						} else {
							j_adj.push_pop(d,i);
						}
					}
				});
			});
		}
	}
	#[allow(unused)]
	fn heuristic_post_prune<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let (n_data, max_cos, dist_is_sq) = (self.n_data, F::from(0.55).unwrap(), true);
		let n_threads = self.n_threads;
		let dist_fun = |a,b| self._get_dist(mat,a,b);
		let unsafe_self_ref = std::ptr::from_ref(self) as *mut Self;
		let graphs = unsafe{&mut (*unsafe_self_ref).graphs};
		graphs.iter_mut()
		.enumerate()
		.for_each(|(i_layer, graph)| {
			let n_nodes = graph.n_vertices();
			let thread_chunk_size = (n_nodes+n_threads-1)/n_threads;
			let global_ids = if i_layer > 0 { Some(&self.global_layer_ids[i_layer-1]) } else { None };

			(0..n_nodes).step_by(thread_chunk_size)
			.map(|start| start..(start+thread_chunk_size).min(n_nodes))
			.par_bridge()
			.for_each(|chunk| {
				let unsafe_graph_ref = std::ptr::from_ref(graph) as *mut HNSWHeapBuildGraph<R,F>;
			
				let mut old_neighbors: Vec<(F,R)> = vec![];
				unsafe{
					chunk.for_each(|i| {
						let (iusize, i) = (i, R::from_usize(i).unwrap_unchecked());
						let i_adj: &mut MaxHeap<F,R> = (*unsafe_graph_ref).view_neighbors_heap_mut(i);

						old_neighbors.clear();
						let n_elem = i_adj.size();
						old_neighbors.reserve(n_elem);
						old_neighbors.set_len(n_elem);
						old_neighbors.iter_mut().rev().zip(i_adj.sorted_iter()).for_each(|(x,y)| *x = y);
		
						old_neighbors.iter().for_each(|&(ijdist, j)| {
							let jusize = R::to_usize(&j).unwrap_unchecked();
							let jglobal = if global_ids.is_some() { global_ids.unwrap_unchecked()[jusize].to_usize().unwrap_unchecked() } else {jusize};
							let mut keep_neighbor = true;
							for &(ikdist, k) in i_adj.iter() {
								let kusize = R::to_usize(&k).unwrap_unchecked();
								let kglobal = if global_ids.is_some() { global_ids.unwrap_unchecked()[kusize].to_usize().unwrap_unchecked() } else {kusize};
								let jkdist: F = dist_fun(jglobal, kglobal);
								if jkdist < ikdist {
									keep_neighbor = false;
									break
								}
							}
							if keep_neighbor {
								i_adj.push(ijdist,j);
							}
						});
					});
				}
			})
		});
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for HNSWParallelHeapBuilder<R, F, Dist> {
	type Params = HNSWParams<F>;
	type Graph = HNSWHeapBuildGraph<R,F>;
	#[inline(always)]
	fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
	#[inline(always)]
	fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
	#[inline(always)]
	fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
	#[inline(always)]
	fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
	#[inline(always)]
	fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
		(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
	}
	#[inline(always)]
	fn _max_degrees(&self) -> (usize, usize) {
		(self.params.lowest_max_degree, self.params.higher_max_degree)
	}
	fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let n_threads = current_num_threads();
		let level_norm_param = params.level_norm_param_override.unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
		/* By the law of large numbers, this is the appropriate number of layers */
		let n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1).min(params.max_layers);
		let mut builder = Self {
			_phantom: std::marker::PhantomData,
			n_data,
			n_threads,
			params,
			node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
			n_layers,
			graphs: (0..n_layers).map(|_| HNSWHeapBuildGraph::new()).collect(),
			local_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			global_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			dist,
			level_norm_param,
		};
		builder.train(mat);
		builder
	}

}



pub struct FloodingHNSWBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	params: HNSWParams<F>,
	graphs: Vec<HNSWHeapBuildGraph<R,F>>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	dist: Dist,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> FloodingHNSWBuilder<R, F, Dist> {
	fn flood_select<G: Graph<R>+Sync>(graph: &G, max_degree: usize) -> Vec<R> {
		let n_vertices = graph.n_vertices();
		let mut in_edges: Vec<Vec<R>> = vec![Vec::with_capacity(max_degree); n_vertices];
		unsafe {
			(0..n_vertices).for_each(|i| {
				let i = R::from_usize(i).unwrap_unchecked();
				graph.iter_neighbors(i).for_each(|j| {
					in_edges[j.to_usize().unwrap_unchecked()].push(i);
				});
			});
		}
		let n_buckets = (n_vertices+63)/64;
		let selected = vec![0u64; n_buckets];
		let perm = RandomPermutationGenerator::new(n_vertices, 4);
		unsafe {
			let n_threads = rayon::current_num_threads();
			let chunk_size = (n_vertices + n_threads - 1) / n_threads;
			(0..n_threads).into_par_iter().map(|i_chunk| {
				let start = chunk_size * i_chunk;
				let end = (start+chunk_size).min(n_vertices);
				let mut selected_idxs = Vec::with_capacity(chunk_size);
				let unsafe_selected = std::ptr::addr_of!(selected) as *const Vec<u64> as *mut Vec<u64>;
				(start..end).for_each(|i| {
					let i = perm.apply_rounds(i);
					let covered = in_edges.get_unchecked(i).iter().any(|j| {
						selected.get_bit_unchecked(j.to_usize().unwrap_unchecked()/64)
					});
					if !covered {
						unsafe_selected.as_mut().unwrap_unchecked().set_bit_unchecked(i/64, true);
						selected_idxs.push(R::from(i).unwrap_unchecked());
					}
				});
				selected_idxs
			}).collect::<Vec<Vec<R>>>().into_iter()
			.map(|v| v.into_iter())
			.flatten().collect()
		}
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let param_max_layers = self.params.max_layers;
		self.params.max_layers = 1;
		let lowest_level_builder: HNSWParallelHeapBuilder<R, F, Dist> = HNSWParallelHeapBuilder::base_init(mat, self.dist.clone(), self.params.clone());
		let (mut graphs,_,_,_) = lowest_level_builder._into_parts();
		self.graphs.push(graphs.pop().unwrap());
		unsafe {
			for i_layer in 1..param_max_layers {
				if self.graphs.last().unwrap().n_vertices() < self.params.higher_max_degree { break; }
				let local_ids: Vec<R> = Self::flood_select(&self.graphs[i_layer-1], self.params.lowest_max_degree);
				let global_ids: Vec<R> = if i_layer==1 {
					local_ids.iter().cloned().collect()
				} else {
					local_ids.iter().map(|&i| {
						self.global_layer_ids[i_layer-2][i.to_usize().unwrap_unchecked()]
					}).collect()
				};
				self.local_layer_ids.push(local_ids);
				self.global_layer_ids.push(global_ids);
				let local_data = mat.get_rows(&self.global_layer_ids[i_layer-1].iter().map(|v| v.to_usize().unwrap_unchecked()).collect::<Vec<_>>());
				self.params.lowest_max_degree = self.params.higher_max_degree;
				let curr_level_builder = HNSWParallelHeapBuilder::base_init(&local_data, self.dist.clone(), self.params.clone());
				let (mut curr_graphs, _, _, _) = curr_level_builder._into_parts();
				self.graphs.push(curr_graphs.pop().unwrap());
			}
		}
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for FloodingHNSWBuilder<R, F, Dist> {
	type Params = HNSWParams<F>;
	type Graph = HNSWHeapBuildGraph<R,F>;
	#[inline(always)]
	fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
	#[inline(always)]
	fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
	#[inline(always)]
	fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
	#[inline(always)]
	fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
	#[inline(always)]
	fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
		(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
	}
	#[inline(always)]
	fn _max_degrees(&self) -> (usize, usize) {
		(self.params.lowest_max_degree, self.params.higher_max_degree)
	}
	fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let mut builder = Self {
			_phantom: std::marker::PhantomData,
			params,
			graphs: vec![],
			local_layer_ids: vec![],
			global_layer_ids: vec![],
			dist,
		};
		builder.train(mat);
		builder
	}

}

pub struct FloodingHNSWSENBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	params: HNSWSENParams<F>,
	graphs: Vec<HNSWHeapBuildGraph<R,F>>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	dist: Dist,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> FloodingHNSWSENBuilder<R, F, Dist> {
	fn flood_select<G: Graph<R>+Sync>(graph: &G, max_degree: usize) -> Vec<R> {
		let n_vertices = graph.n_vertices();
		let mut in_edges: Vec<Vec<R>> = vec![Vec::with_capacity(max_degree); n_vertices];
		unsafe {
			(0..n_vertices).for_each(|i| {
				let i = R::from_usize(i).unwrap_unchecked();
				graph.iter_neighbors(i).for_each(|j| {
					in_edges[j.to_usize().unwrap_unchecked()].push(i);
				});
			});
		}
		let n_buckets = (n_vertices+63)/64;
		let selected = vec![0u64; n_buckets];
		let perm = RandomPermutationGenerator::new(n_vertices, 4);
		unsafe {
			let n_threads = rayon::current_num_threads();
			let chunk_size = (n_vertices + n_threads - 1) / n_threads;
			(0..n_threads).into_par_iter().map(|i_chunk| {
				let start = chunk_size * i_chunk;
				let end = (start+chunk_size).min(n_vertices);
				let mut selected_idxs = Vec::with_capacity(chunk_size);
				let unsafe_selected = std::ptr::addr_of!(selected) as *const Vec<u64> as *mut Vec<u64>;
				(start..end).for_each(|i| {
					let i = perm.apply_rounds(i);
					let covered = in_edges.get_unchecked(i).iter().any(|j| {
						selected.get_bit_unchecked(j.to_usize().unwrap_unchecked()/64)
					});
					if !covered {
						unsafe_selected.as_mut().unwrap_unchecked().set_bit_unchecked(i/64, true);
						selected_idxs.push(R::from(i).unwrap_unchecked());
					}
				});
				selected_idxs
			}).collect::<Vec<Vec<R>>>().into_iter()
			.map(|v| v.into_iter())
			.flatten().collect()
		}
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let param_max_layers = self.params.max_layers;
		self.params.max_layers = 1;
		let lowest_level_builder: HNSWParallelSENHeapBuilder<R, F, Dist> = HNSWParallelSENHeapBuilder::base_init(mat, self.dist.clone(), self.params.clone());
		let (mut graphs,_,_,_) = lowest_level_builder._into_parts();
		self.graphs.push(graphs.pop().unwrap());
		unsafe {
			for i_layer in 1..param_max_layers {
				if self.graphs.last().unwrap().n_vertices() < self.params.higher_max_degree { break; }
				let local_ids: Vec<R> = Self::flood_select(&self.graphs[i_layer-1], self.params.lowest_max_degree);
				let global_ids: Vec<R> = if i_layer==1 {
					local_ids.iter().cloned().collect()
				} else {
					local_ids.iter().map(|&i| {
						self.global_layer_ids[i_layer-2][i.to_usize().unwrap_unchecked()]
					}).collect()
				};
				self.local_layer_ids.push(local_ids);
				self.global_layer_ids.push(global_ids);
				let local_data = mat.get_rows(&self.global_layer_ids[i_layer-1].iter().map(|v| v.to_usize().unwrap_unchecked()).collect::<Vec<_>>());
				self.params.lowest_max_degree = self.params.higher_max_degree;
				let curr_level_builder = HNSWParallelSENHeapBuilder::base_init(&local_data, self.dist.clone(), self.params.clone());
				let (mut curr_graphs, _, _, _) = curr_level_builder._into_parts();
				self.graphs.push(curr_graphs.pop().unwrap());
			}
		}
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for FloodingHNSWSENBuilder<R, F, Dist> {
	type Params = HNSWSENParams<F>;
	type Graph = HNSWHeapBuildGraph<R,F>;
	#[inline(always)]
	fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
	#[inline(always)]
	fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
	#[inline(always)]
	fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
	#[inline(always)]
	fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
	#[inline(always)]
	fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
		(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
	}
	#[inline(always)]
	fn _max_degrees(&self) -> (usize, usize) {
		(self.params.lowest_max_degree, self.params.higher_max_degree)
	}
	fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let mut builder = Self {
			_phantom: std::marker::PhantomData,
			params,
			graphs: vec![],
			local_layer_ids: vec![],
			global_layer_ids: vec![],
			dist,
		};
		builder.train(mat);
		builder
	}

}





param_struct!(HNSWSENParams[Copy, Clone]<F: SyncFloat> {
	higher_max_degree: usize = 50,
	lowest_max_degree: usize = 100,
	max_layers: usize = 10,
	n_parallel_burnin: usize = 0,
	max_build_heap_size: usize = 50,
	max_build_frontier_size: Option<usize> = Some(100),
	level_norm_param_override: Option<f32> = None,
	insert_heuristic: bool = true,
	insert_heuristic_extend: bool = true,
	post_prune_heuristic: bool = false,
	insert_minibatch_size: usize = 100,
	n_rounds: usize = 1,
	finetune_rnn: bool = false,
	finetune_sen: bool = false,
	finetune_rnn_params: RNNParams = RNNParams::new(),
	finetune_sen_params: SENParams<F> = SENParams::new(),
	max_cos: f64 = 0.5,
});
pub struct HNSWParallelSENHeapBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	n_threads: usize,
	params: HNSWSENParams<F>,
	node_locks: Vec<Mutex<()>>,
	n_layers: usize,
	graphs: Vec<HNSWHeapBuildGraph<R,F>>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	dist: Dist,
	level_norm_param: f32,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWParallelSENHeapBuilder<R, F, Dist> {
	fn initialize_structure(&mut self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
		/* Precompute all point levels */
		let layer_sizes = (0..self.n_layers).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
		let mut point_levels = Vec::with_capacity(self.n_data);
		unsafe{point_levels.set_len(self.n_data)};
		let mut point_level_pos: Vec<usize> = Vec::with_capacity(self.n_data);
		unsafe{point_level_pos.set_len(self.n_data)};
		/* Always insert first point on highest level */
		point_levels[0] = self.n_layers-1;
		layer_sizes[self.n_layers-1].store(1, Relaxed);
		/* Insert all other points */
		let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
		point_levels[1..].chunks_mut(chunk_size).par_bridge().for_each(|point_level_chunk| {
			point_level_chunk.iter_mut().for_each(|point_level_ref| {
				let level = random_level(self.level_norm_param, self.n_layers);
				*point_level_ref = level;
				layer_sizes[level].fetch_add(1, Relaxed);
			})
		});
		/* Remove atomic wrappers */
		let mut layer_sizes = layer_sizes.into_iter().map(|x| x.into_inner()).collect::<Vec<_>>();
		/* Aggregate layer sizes such that the bottom layer has size N */
		(0..self.n_layers-1).rev().for_each(|i| layer_sizes[i] += layer_sizes[i+1]);
		debug_assert!(layer_sizes[0] == self.n_data);
		/* Initialize lowest level */
		let graph0 = &mut self.graphs[0];
		graph0.reserve(self.n_data);
		(0..self.n_data).for_each(|_| graph0.add_node_with_capacity(self.params.lowest_max_degree));
		/* Initialize all other levels */
		unsafe {
			/* Preallocate graphs and ID lookup memory (warning: uninitialized RAM, handle with care!) */
			layer_sizes.iter().skip(1)
			.zip(self.local_layer_ids.iter_mut())
			.zip(self.global_layer_ids.iter_mut())
			.zip(self.graphs.iter_mut().skip(1))
			.par_bridge()
			.for_each(|(((&layer_size, local_ids), global_ids), graph)| {
				local_ids.reserve(layer_size);
				local_ids.set_len(layer_size);
				global_ids.reserve(layer_size);
				global_ids.set_len(layer_size);
				graph.reserve(layer_size);
				(0..layer_size).for_each(|_| graph.add_node_with_capacity(self.params.higher_max_degree));
			});
			/* Set level pos for layer 0 */
			point_level_pos.chunks_mut(chunk_size)
			.zip(point_levels.chunks(chunk_size))
			.enumerate().par_bridge().for_each(|(i_chunk, (point_level_pos_chunk, point_level_chunk))| {
				point_level_pos_chunk.iter_mut().zip(point_level_chunk.iter()).enumerate()
				.filter(|(_,(_,&level))| level==0).for_each(|(i, (point_level_pos_ref, _))| {
					*point_level_pos_ref = i_chunk*chunk_size + i;
				});
			});
			/* Set level pos for all higher levels */
			self.local_layer_ids.iter_mut()
			.zip(self.global_layer_ids.iter_mut())
			.enumerate().par_bridge().for_each(|(i_layer,(local_ids, global_ids))| {
				/* Unsafe reference for (unsafe) parallel write access */
				let unsafe_point_level_pos = std::ptr::addr_of!(point_level_pos) as *mut Vec<usize>;
				/* Incrementing enumerator as the ID arrays start at layer 1 */
				let i_layer = i_layer+1;
				/* Counting local IDs for current and next layer to properly connect them in ID lookups */
				let next_level = i_layer-1;
				let curr_level = i_layer;
				let mut next_layer_offset = R::zero();
				let mut curr_layer_offset = 0;
				/* Setting up local and global ID lookups */
				point_levels.iter().enumerate().for_each(|(i, &level)| {
					if level >= next_level {
						/* Point i requires an entry in the lookups of the current layer */
						if level >= curr_level {
							local_ids[curr_layer_offset] = next_layer_offset;
							global_ids[curr_layer_offset] = R::from_usize(i).unwrap_unchecked();
							if level == curr_level { (&mut *unsafe_point_level_pos)[i] = curr_layer_offset; }
							curr_layer_offset += 1;
						}
						/* Point i requires an entry in the lookups of the next layer */
						next_layer_offset += R::one();
					}
				});
				/* Ensure that all IDs between 0 and N were used for this and next layer */
				debug_assert!(curr_layer_offset == layer_sizes[curr_level]);
				debug_assert!(next_layer_offset.to_usize().unwrap() == layer_sizes[next_level]);
			});
			#[cfg(debug_assertions)]
			{
				point_levels.iter().zip(point_level_pos.iter()).enumerate().for_each(|(i,(&level, &pos))| {
					assert!(layer_sizes[level] > pos);
					let mut pos = pos;
					(1..level+1).rev().for_each(|i_layer| {
						assert_eq!(i, self.global_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked());
						pos = self.local_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked();
					});
					assert_eq!(i, pos);
				});
			}
		}
		(layer_sizes, point_levels, point_level_pos)
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		/* Do nothing for empty datasets */
		if self.n_data == 0 { return; }
		/* Preallocate graphs and ID lookups and assign levels and level positions for each input */
		let (layer_sizes, point_levels, point_level_pos) = self.initialize_structure();
		/* Burnin single thread for a few samples */
		let n_burnin: usize = self.params.n_parallel_burnin.min(self.n_data-1);
		let mut burnin_cache = HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size);
		(1..n_burnin+1).for_each(|i| {
			self.insert(0, mat, point_level_pos[i], i, point_levels[i], &mut burnin_cache);
		});
		/* Insert edges to the graphs as per HNSW insertion rules */
		let mut thread_caches = (0..self.n_threads).map(|_| HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size)).collect::<Vec<_>>();
		let chunk_size = (self.n_data-n_burnin + self.n_threads - 1) / self.n_threads;
		(1+n_burnin..self.n_data).step_by(chunk_size)
		.map(|start| start..(start+chunk_size).min(self.n_data))
		.zip(thread_caches.iter_mut())
		.par_bridge().for_each(|(chunk, thread_cache)| {
			let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
			unsafe{
				chunk.for_each(|i| {
					(*unsafe_self_ref).insert(0, mat, point_level_pos[i], i, point_levels[i], thread_cache);
				});
			}
		});
		(1..self.params.n_rounds).for_each(|i_round| {
			let perm_gen = RandomPermutationGenerator::new(self.n_data-1, 4);
			let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
			(0..self.n_data-1).step_by(chunk_size)
			.map(|start| start..(start+chunk_size).min(self.n_data-1))
			.zip(thread_caches.iter_mut())
			.par_bridge().for_each(|(chunk, thread_cache)| {
				let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
				unsafe{
					chunk.for_each(|i| {
						let i = perm_gen.apply_rounds(i)+1;
						(*unsafe_self_ref).insert(i_round, mat, point_level_pos[i], i, point_levels[i], thread_cache);
					});
				}
			});
		});
		if self.params.finetune_rnn {
			self.finetune_rnn(mat, self.params.finetune_rnn_params.clone());
		}
		if self.params.finetune_sen {
			self.finetune_sen(mat, self.params.finetune_sen_params.clone());
		}
		if self.params.post_prune_heuristic {
			self.prune_non_sen_edges(mat);
		}
	}
	fn insert<M: MatrixDataSource<F>+Sync>(&mut self, _i_round: usize, mat: &M, mut i: usize, i_global: usize, level: usize, thread_cache: &mut HNSWThreadCache<R,F>) {
		/* We just assume that the global index 1 is the root entry point */
		let top_entry_dist = self._get_dist(mat, i_global, 0);
		thread_cache.entry_points.clear();
		thread_cache.entry_points.push((top_entry_dist, R::zero()));
		/* Search entry point for the required level */
		(level+1..self.n_layers).rev().for_each(|i_layer| {
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, None);
			let local_ids = &self.local_layer_ids[i_layer-1];
			thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
		});
		(0..level+1).rev().for_each(|i_layer| {
			// if i_round > 0 {self.graphs[i_layer].view_neighbors_heap_mut(unsafe{R::from_usize(i).unwrap_unchecked()}).clear();}
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, Some(self.params.max_build_heap_size));
			self.insert_layer(mat, i, i_global, i_layer, thread_cache);
			if i_layer > 0 {
				let local_ids = &self.local_layer_ids[i_layer-1];
				thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
				i = unsafe{local_ids[i].to_usize().unwrap_unchecked()};
			}
		});
	}
	fn insert_layer<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, i_global: usize, layer: usize, thread_cache: &mut HNSWThreadCache<R,F>) {
		/* Choose the appropriate max degree */
		let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
		/* TODO: Additional parameters should eventually be moved to a new parameter type */
		let max_cos = F::from(self.params.max_cos).unwrap();
		let min_neighbors = max_neighbors;
		// let min_neighbors = 20;
		// let min_neighbors = 0;
		// let protected_neighbors = max_neighbors/2;
		// let protected_neighbors = 20;
		let protected_neighbors = 0;
		let neighbors = &mut thread_cache.entry_points;
		debug_assert_eq!(neighbors.len(), neighbors.iter().map(|(_,j)| j).collect::<std::collections::HashSet<_>>().len());
		let heuristic_hashset = unsafe{thread_cache.heuristic_hashsets.get_unchecked_mut(layer)};
		/* Translate i into a local (current graph) and a global (dataset/distance computations) ID */
		let i = unsafe{R::from_usize(i).unwrap_unchecked()};
		/* Choose the appropriate max degree */
		let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
		/* Unsafe graph reference to have `self` accessible downstream */
		let graph = &mut self.graphs[layer] as *mut HNSWHeapBuildGraph<R,F>;
		unsafe {
			let i_adj = (*graph).view_neighbors_heap_mut(i);
			/* Reduce neighbors to set of max_neighbors points by using a heuristic */
			/* Then bidirectionally link i to all of these neighbors */
			/* Afterwards prune neighborhoods of all affected points back down to max_neighbors */
			if !self.params.insert_heuristic {
				heuristic_hashset.clear();
				heuristic_hashset.insert(i);
				with_guard!(self, i_global, {
					i_adj.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					neighbors.iter()
					/* This should almost never matter, but this node can be found by another
					 * searching thread while creating the adjacency list. */
					.filter(|&&(_,j)| heuristic_hashset.insert(j))
					.take(max_neighbors)
					.for_each(|&(d,j)| {
						if i_adj.size() < max_neighbors {
							i_adj.push(d,j);
						} else {
							i_adj.push_pop(d,j);
						}
					});
				});
			} else {
				/* Maybe extend candidate set with neighbors of neighbors */
				let mut candidates = neighbors.clone();
				/* Add previous neighbors to the candidate set */
				candidates.extend(i_adj.iter());
				i_adj.clear();
				if self.params.insert_heuristic_extend {
					heuristic_hashset.clear();
					candidates.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					(0..neighbors.len()).for_each(|i_neighbor| {
						(*graph).view_neighbors(R::from_usize(i_neighbor).unwrap_unchecked()).iter().for_each(|&(_,j)| {
							if heuristic_hashset.insert(j) {
								let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
								candidates.push((self._get_dist(mat, i_global, j_global.to_usize().unwrap_unchecked()), j));
							}
						});
					});
				}
				/* Make sure that the candidates are truly sorted and unique */
				candidates.sort_unstable_by(|a,b| {
					let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
					if dist_cmp.is_eq() { a.1.cmp(&b.1) } else { dist_cmp }
				});
				remove_duplicates_with_key(&mut candidates, |(_,j)| j);
				/* Prune neighborhoods with relative neighbor heuristic */
				heuristic_hashset.clear();
				heuristic_hashset.insert(i);
				let mut tmp_list = Vec::with_capacity(candidates.len());
				let mut cand_iter = candidates.into_iter().peekable();
				with_guard!(self, i_global, {
					while cand_iter.peek().is_some() && heuristic_hashset.contains(&cand_iter.peek().unwrap_unchecked().1) { cand_iter.next(); }
					if cand_iter.peek().is_some() {
						let (dij0,j0) = cand_iter.next().unwrap_unchecked();
						i_adj.push(dij0,j0);
						heuristic_hashset.insert(j0);
						for (dij,j) in cand_iter {
							if i_adj.size() >= max_neighbors { break; }
							if !heuristic_hashset.insert(j) { continue; }
							if i_adj.iter().all(|&(dik,k)| {
								let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
								let k_global = if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k };
								let djk = self._get_dist(mat, j_global.to_usize().unwrap_unchecked(), k_global.to_usize().unwrap_unchecked());
								// dik < djk
								/* Only insert if angle to all neighbors is larger than threshold (cos is smaller than threshold) */
								tri_to_cos::<F>(dij,dik,djk,true) < max_cos
							}) {
								i_adj.push(dij,j);
							} else {
								tmp_list.push((dij,j));
							}
						};
						if max_neighbors > i_adj.size() {
							/* Add removed edges back in */
							tmp_list.iter().take(max_neighbors-i_adj.size()).for_each(|&x| i_adj.push(x.0,x.1));
						}
					}
				});
			}
			/* Add backward edges */
			let mut tmp_list = Vec::new();
			let mut tmp_rem_list = Vec::new();
			i_adj.iter().for_each(|&(dij,j)| {
				let j_adj = (*graph).view_neighbors_heap_mut(j);
				let j_global = (if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j }).to_usize().unwrap_unchecked();
				/* Either replace maximum are append */
				with_guard!(self, j_global, {
					/* Due to race conditions we must ensure, that i is not in this adjacency list */
					if (j_adj.size()==0 || j_adj.peek().unwrap_unchecked().0 > dij) && j_adj.iter().all(|&(_,k)| k != i) {
						if j_adj.size() < max_neighbors {
							/* Only insert into smaller adj list if angle to all previous neighbors is larger enough (cos is small enough) */
							if j_adj.size() < min_neighbors || j_adj.iter().all(|&(djk,k)| djk < dij || {
								let k_global = (if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k }).to_usize().unwrap_unchecked();
								let dik = self._get_dist(mat, i_global, k_global);
								tri_to_cos(dij,djk,dik,true) < max_cos
							}) {
								j_adj.push(dij,i);
							}
						} else {
							if false {
								tmp_list.clear();
								tmp_rem_list.clear();
								loop {
									/* Replace max distance if larger than dij and no heuristic replacement was found */
									if j_adj.size() <= protected_neighbors || j_adj.peek().unwrap_unchecked().0 < dij {
										let n = tmp_list.len();
										if n > 0 {
											tmp_list.push((dij,i));
										}
										break;
									}
									/* Test heuristic replacement */
									let (djk,k) = j_adj.pop().unwrap_unchecked();
									let k_global = (if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k }).to_usize().unwrap_unchecked();
									let dik = self._get_dist(mat, i_global, k_global);
									/* If heuristic criterion applies, remove neighbor */
									/* I.e. only keep neighbors with an angle larger than threshold (cos smaller than threshold) */
									if tri_to_cos(dij,djk,dik,true) < max_cos {
										tmp_list.push((djk,k));
									} else {
										tmp_rem_list.push((djk,k));
									}
								}
								tmp_list.iter().skip((j_adj.size()+tmp_list.len()).max(max_neighbors)-max_neighbors).for_each(|&(djk,k)| j_adj.push(djk,k));
								tmp_rem_list.iter().rev().take(min_neighbors-j_adj.size().min(min_neighbors)).for_each(|&x| j_adj.push(x.0,x.1));
								// assert_eq!(j_adj.size(), max_neighbors);
							} else {
								/* Just replace the maximum distance entry if it is dominated */
								j_adj.push_pop(dij,i);
							}
						}
					}
				});
			});
		}
	}
	#[allow(unused)]
	fn prune_non_sen_edges<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		let (n_data, max_cos, dist_is_sq) = (self.n_data, F::from(self.params.max_cos).unwrap(), true);
		let n_threads = self.n_threads;
		let dist_fun = |a,b| self._get_dist(mat,a,b);
		let unsafe_self_ref = std::ptr::from_ref(self) as *mut Self;
		let graphs = unsafe{&mut (*unsafe_self_ref).graphs};
		graphs.iter_mut()
		.enumerate()
		.for_each(|(i_layer, graph)| {
			let n_nodes = graph.n_vertices();
			let thread_chunk_size = (n_nodes+n_threads-1)/n_threads;
			let global_ids = if i_layer > 0 { Some(&self.global_layer_ids[i_layer-1]) } else { None };

			(0..n_nodes).step_by(thread_chunk_size)
			.map(|start| start..(start+thread_chunk_size).min(n_nodes))
			.par_bridge()
			.for_each(|chunk| {
				let unsafe_graph_ref = std::ptr::from_ref(graph) as *mut HNSWHeapBuildGraph<R,F>;
			
				let mut old_neighbors: Vec<(F,R)> = vec![];
				unsafe{
					chunk.for_each(|i| {
						let (iusize, i) = (i, R::from_usize(i).unwrap_unchecked());
						let i_adj: &mut MaxHeap<F,R> = (*unsafe_graph_ref).view_neighbors_heap_mut(i);

						old_neighbors.clear();
						let n_elem = i_adj.size();
						old_neighbors.reserve(n_elem);
						old_neighbors.set_len(n_elem);
						old_neighbors.iter_mut().rev().zip(i_adj.sorted_iter()).for_each(|(x,y)| *x = y);
		
						old_neighbors.iter().for_each(|&(ijdist, j)| {
							let jusize = R::to_usize(&j).unwrap_unchecked();
							let jglobal = if global_ids.is_some() { global_ids.unwrap_unchecked()[jusize].to_usize().unwrap_unchecked() } else {jusize};
							let mut keep_neighbor = true;
							for &(ikdist, k) in i_adj.iter() {
								let kusize = R::to_usize(&k).unwrap_unchecked();
								let kglobal = if global_ids.is_some() { global_ids.unwrap_unchecked()[kusize].to_usize().unwrap_unchecked() } else {kusize};
								let jkdist: F = dist_fun(jglobal, kglobal);
								let cos = tri_to_cos(ijdist,ikdist,jkdist,dist_is_sq);
								if cos >= max_cos {
									keep_neighbor = false;
									break
								}
							}
							if keep_neighbor {
								i_adj.push(ijdist,j);
							}
						});
					});
				}
			})
		});
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for HNSWParallelSENHeapBuilder<R, F, Dist> {
	type Params = HNSWSENParams<F>;
	type Graph = HNSWHeapBuildGraph<R,F>;
	#[inline(always)]
	fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
	#[inline(always)]
	fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
	#[inline(always)]
	fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
	#[inline(always)]
	fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
	#[inline(always)]
	fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
		(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
	}
	#[inline(always)]
	fn _max_degrees(&self) -> (usize, usize) {
		(self.params.lowest_max_degree, self.params.higher_max_degree)
	}
	fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let n_threads = current_num_threads();
		let level_norm_param = params.level_norm_param_override.unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
		/* By the law of large numbers, this is the appropriate number of layers */
		let n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1).min(params.max_layers);
		let mut builder = Self {
			_phantom: std::marker::PhantomData,
			n_data,
			n_threads,
			params,
			node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
			n_layers,
			graphs: (0..n_layers).map(|_| HNSWHeapBuildGraph::new()).collect(),
			local_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			global_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			dist,
			level_norm_param,
		};
		builder.train(mat);
		builder
	}

}



pub struct HNSWParallelPresortedHeapBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	n_threads: usize,
	params: HNSWParams<F>,
	node_locks: Vec<Mutex<()>>,
	n_layers: usize,
	graphs: Vec<HNSWHeapBuildGraph<R,F>>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	dist: Dist,
	level_norm_param: f32,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWParallelPresortedHeapBuilder<R, F, Dist> {
	fn initialize_structure(&mut self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
		/* Precompute all point levels */
		let layer_sizes = (0..self.n_layers).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
		let mut point_levels = Vec::with_capacity(self.n_data);
		unsafe{point_levels.set_len(self.n_data)};
		let mut point_level_pos: Vec<usize> = Vec::with_capacity(self.n_data);
		unsafe{point_level_pos.set_len(self.n_data)};
		/* Always insert first point on highest level */
		point_levels[0] = self.n_layers-1;
		layer_sizes[self.n_layers-1].store(1, Relaxed);
		/* Insert all other points */
		let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
		point_levels[1..].chunks_mut(chunk_size).par_bridge().for_each(|point_level_chunk| {
			point_level_chunk.iter_mut().for_each(|point_level_ref| {
				let level = random_level(self.level_norm_param, self.n_layers);
				*point_level_ref = level;
				layer_sizes[level].fetch_add(1, Relaxed);
			})
		});
		/* Remove atomic wrappers */
		let mut layer_sizes = layer_sizes.into_iter().map(|x| x.into_inner()).collect::<Vec<_>>();
		/* Aggregate layer sizes such that the bottom layer has size N */
		(0..self.n_layers-1).rev().for_each(|i| layer_sizes[i] += layer_sizes[i+1]);
		debug_assert!(layer_sizes[0] == self.n_data);
		/* Initialize lowest level */
		let graph0 = &mut self.graphs[0];
		graph0.reserve(self.n_data);
		(0..self.n_data).for_each(|_| graph0.add_node_with_capacity(self.params.lowest_max_degree));
		/* Initialize all other levels */
		unsafe {
			/* Preallocate graphs and ID lookup memory (warning: uninitialized RAM, handle with care!) */
			layer_sizes.iter().skip(1)
			.zip(self.local_layer_ids.iter_mut())
			.zip(self.global_layer_ids.iter_mut())
			.zip(self.graphs.iter_mut().skip(1))
			.par_bridge()
			.for_each(|(((&layer_size, local_ids), global_ids), graph)| {
				local_ids.reserve(layer_size);
				local_ids.set_len(layer_size);
				global_ids.reserve(layer_size);
				global_ids.set_len(layer_size);
				graph.reserve(layer_size);
				(0..layer_size).for_each(|_| graph.add_node_with_capacity(self.params.higher_max_degree));
			});
			/* Set level pos for layer 0 */
			point_level_pos.chunks_mut(chunk_size)
			.zip(point_levels.chunks(chunk_size))
			.enumerate().par_bridge().for_each(|(i_chunk, (point_level_pos_chunk, point_level_chunk))| {
				point_level_pos_chunk.iter_mut().zip(point_level_chunk.iter()).enumerate()
				.filter(|(_,(_,&level))| level==0).for_each(|(i, (point_level_pos_ref, _))| {
					*point_level_pos_ref = i_chunk*chunk_size + i;
				});
			});
			/* Set level pos for all higher levels */
			self.local_layer_ids.iter_mut()
			.zip(self.global_layer_ids.iter_mut())
			.enumerate().par_bridge().for_each(|(i_layer,(local_ids, global_ids))| {
				/* Unsafe reference for (unsafe) parallel write access */
				let unsafe_point_level_pos = std::ptr::addr_of!(point_level_pos) as *mut Vec<usize>;
				/* Incrementing enumerator as the ID arrays start at layer 1 */
				let i_layer = i_layer+1;
				/* Counting local IDs for current and next layer to properly connect them in ID lookups */
				let next_level = i_layer-1;
				let curr_level = i_layer;
				let mut next_layer_offset = R::zero();
				let mut curr_layer_offset = 0;
				/* Setting up local and global ID lookups */
				point_levels.iter().enumerate().for_each(|(i, &level)| {
					if level >= next_level {
						/* Point i requires an entry in the lookups of the current layer */
						if level >= curr_level {
							local_ids[curr_layer_offset] = next_layer_offset;
							global_ids[curr_layer_offset] = R::from_usize(i).unwrap_unchecked();
							if level == curr_level { (&mut *unsafe_point_level_pos)[i] = curr_layer_offset; }
							curr_layer_offset += 1;
						}
						/* Point i requires an entry in the lookups of the next layer */
						next_layer_offset += R::one();
					}
				});
				/* Ensure that all IDs between 0 and N were used for this and next layer */
				debug_assert!(curr_layer_offset == layer_sizes[curr_level]);
				debug_assert!(next_layer_offset.to_usize().unwrap() == layer_sizes[next_level]);
			});
			#[cfg(debug_assertions)]
			{
				point_levels.iter().zip(point_level_pos.iter()).enumerate().for_each(|(i,(&level, &pos))| {
					assert!(layer_sizes[level] > pos);
					let mut pos = pos;
					(1..level+1).rev().for_each(|i_layer| {
						assert_eq!(i, self.global_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked());
						pos = self.local_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked();
					});
					assert_eq!(i, pos);
				});
			}
		}
		(layer_sizes, point_levels, point_level_pos)
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		/* Do nothing for empty datasets */
		if self.n_data == 0 { return; }
		/* Preallocate graphs and ID lookups and assign levels and level positions for each input */
		let (layer_sizes, point_levels, point_level_pos) = self.initialize_structure();
		let mut order = (1..self.n_data).collect::<Vec<_>>();
		order.par_sort_unstable_by(|&i,&j| point_levels[j].cmp(&point_levels[i]));
		println!("{:?}", order[0..50].iter().map(|&i| (i,point_levels[i])).collect::<Vec<_>>());
		/* Burnin single thread for a few samples */
		let n_burnin = self.params.n_parallel_burnin;
		let mut burnin_cache = HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size);
		(0..n_burnin).for_each(|i| {
			let i = order[i];
			self.insert(mat, point_level_pos[i], i, point_levels[i], &mut burnin_cache);
		});
		/* Insert edges to the graphs as per HNSW insertion rules */
		let chunk_size = (self.n_data-n_burnin + self.n_threads - 1) / self.n_threads;
		(n_burnin..self.n_data-1).step_by(chunk_size)
		.map(|start| start..(start+chunk_size).min(self.n_data-1))
		.par_bridge().for_each(|chunk| {
			let mut thread_cache = HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size);
			let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
			#[cfg(debug_assertions)]
			unsafe { /* Ensure that the self ref actually works */
				assert_eq!(self.n_data, (*unsafe_self_ref).n_data);
				assert_eq!(self.n_layers, (*unsafe_self_ref).n_layers);
				assert_eq!(self.params.max_build_heap_size, (*unsafe_self_ref).params.max_build_heap_size);
				assert_eq!(self.params.lowest_max_degree, (*unsafe_self_ref).params.lowest_max_degree);
				assert_eq!(self.params.max_build_frontier_size, (*unsafe_self_ref).params.max_build_frontier_size);
				assert_eq!(self.params.insert_heuristic, (*unsafe_self_ref).params.insert_heuristic);
				assert_eq!(self.params.insert_heuristic_extend, (*unsafe_self_ref).params.insert_heuristic_extend);
			}
			unsafe{
				chunk.for_each(|i| {
					let i = order[i];
					(*unsafe_self_ref).insert(mat, point_level_pos[i], i, point_levels[i], &mut thread_cache);
				});
			}
		});
	}
	fn insert<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, mut i: usize, i_global: usize, level: usize, thread_cache: &mut HNSWThreadCache<R,F>) {
		/* We just assume that the global index 1 is the root entry point */
		let top_entry_dist = self._get_dist(mat, i_global, 0);
		thread_cache.entry_points.clear();
		thread_cache.entry_points.push((top_entry_dist, R::zero()));
		/* Search entry point for the required level */
		(level+1..self.n_layers).rev().for_each(|i_layer| {
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, None);
			let local_ids = &self.local_layer_ids[i_layer-1];
			thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
		});
		(0..level+1).rev().for_each(|i_layer| {
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, Some(self.params.max_build_heap_size));
			self.insert_layer(mat, i, i_global, i_layer, thread_cache);
			if i_layer > 0 {
				let local_ids = &self.local_layer_ids[i_layer-1];
				thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
				i = unsafe{local_ids[i].to_usize().unwrap_unchecked()};
			}
		});
	}
	fn insert_layer<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, i_global: usize, layer: usize, thread_cache: &mut HNSWThreadCache<R,F>) {
		let neighbors = &mut thread_cache.entry_points;
		debug_assert_eq!(neighbors.len(), neighbors.iter().map(|(_,j)| j).collect::<std::collections::HashSet<_>>().len());
		let heuristic_hashset = unsafe{thread_cache.heuristic_hashsets.get_unchecked_mut(layer)};
		/* Translate i into a local (current graph) and a global (dataset/distance computations) ID */
		let i = unsafe{R::from_usize(i).unwrap_unchecked()};
		/* Choose the appropriate max degree */
		let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
		/* Unsafe graph reference to have `self` accessible downstream */
		let graph = &mut self.graphs[layer] as *mut HNSWHeapBuildGraph<R,F>;
		unsafe {
			let i_adj = (*graph).view_neighbors_heap_mut(i);
			/* Reduce neighbors to set of max_neighbors points by using a heuristic */
			/* Then bidirectionally link i to all of these neighbors */
			/* Afterwards prune neighborhoods of all affected points back down to max_neighbors */
			if !self.params.insert_heuristic {
				heuristic_hashset.clear();
				heuristic_hashset.insert(i);
				with_guard!(self, i_global, {
					i_adj.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					neighbors.iter()
					/* This should almost never matter, but this node can be found by another
					 * searching thread while creating the adjacency list. */
					.filter(|&&(_,j)| !heuristic_hashset.contains(&j))
					.take(max_neighbors)
					.for_each(|&(d,j)| {
						if i_adj.size() < max_neighbors {
							i_adj.push(d,j);
						} else {
							i_adj.push_pop(d,j);
						}
					});
				});
			} else {
				/* Maybe extend candidate set with neighbors of neighbors */
				let mut candidates = neighbors.clone();
				if self.params.insert_heuristic_extend {
					heuristic_hashset.clear();
					candidates.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					(0..neighbors.len()).for_each(|i_neighbor| {
						(*graph).view_neighbors(R::from_usize(i_neighbor).unwrap_unchecked()).iter().for_each(|&(_,j)| {
							if heuristic_hashset.insert(j) {
								let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
								candidates.push((self._get_dist(mat, i_global, j_global.to_usize().unwrap_unchecked()), j));
							}
						});
					});
				}
				/* Make sure that the candidates are truly sorted and unique */
				candidates.sort_unstable_by(|a,b| {
					let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
					if dist_cmp.is_eq() { a.1.cmp(&b.1) } else { dist_cmp }
				});
				remove_duplicates_with_key(&mut candidates, |(_,j)| j);
				/* Prune neighborhoods with relative neighbor heuristic */
				heuristic_hashset.clear();
				heuristic_hashset.insert(i);
				let mut tmp_list = Vec::with_capacity(candidates.len());
				let mut cand_iter = candidates.into_iter().peekable();
				with_guard!(self, i_global, {
					i_adj.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					while cand_iter.peek().is_some() && heuristic_hashset.contains(&cand_iter.peek().unwrap_unchecked().1) { cand_iter.next(); }
					if cand_iter.peek().is_some() {
						let (dij,j) = cand_iter.next().unwrap_unchecked();
						i_adj.push(dij,j);
						heuristic_hashset.insert(j);
						for (dij,j) in cand_iter {
							if i_adj.size() >= max_neighbors { break; }
							if heuristic_hashset.contains(&j) { continue; }
							if i_adj.iter().all(|&(dik,k)| {
								let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
								let k_global = if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k };
								let djk = self._get_dist(mat, j_global.to_usize().unwrap_unchecked(), k_global.to_usize().unwrap_unchecked());
								dik < djk
								// tri_to_cos(dij.to_f32().unwrap_unchecked(),dik.to_f32().unwrap_unchecked(),djk.to_f32().unwrap_unchecked(),true) < 0.5
							}) {
								i_adj.push(dij,j);
							} else {
								tmp_list.push((dij,j));
							}
						};
						if max_neighbors > i_adj.size() {
							/* Add removed edges back in */
							tmp_list.iter().take(max_neighbors-i_adj.size()).for_each(|&x| i_adj.push(x.0,x.1));
						}
					}
				});
			}
			/* Add backward edges */
			i_adj.iter().for_each(|&(d,j)| {
				let j_adj = (*graph).view_neighbors_heap_mut(j);
				let j_global = (if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j }).to_usize().unwrap_unchecked();
				/* Either replace maximum are append */
				with_guard!(self, j_global, {
					/* Due to race conditions we must ensure, that i is not in this adjacency list */
					if j_adj.iter().all(|&(_,k)| k != i) {
						if j_adj.size() < max_neighbors {
							j_adj.push(d,i);
						} else {
							j_adj.push_pop(d,i);
						}
					}
				});
			});
		}
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for HNSWParallelPresortedHeapBuilder<R, F, Dist> {
	type Params = HNSWParams<F>;
	type Graph = HNSWHeapBuildGraph<R,F>;
	#[inline(always)]
	fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
	#[inline(always)]
	fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
	#[inline(always)]
	fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
	#[inline(always)]
	fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
	#[inline(always)]
	fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
		(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
	}
	#[inline(always)]
	fn _max_degrees(&self) -> (usize, usize) {
		(self.params.lowest_max_degree, self.params.higher_max_degree)
	}
	fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let n_threads = current_num_threads();
		let level_norm_param = params.level_norm_param_override.unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
		/* By the law of large numbers, this is the appropriate number of layers */
		let n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1).min(params.max_layers);
		let mut builder = Self {
			_phantom: std::marker::PhantomData,
			n_data,
			n_threads,
			params,
			node_locks: (0..n_data).map(|_| Mutex::new(())).collect(),
			n_layers,
			graphs: (0..n_layers).map(|_| HNSWHeapBuildGraph::new()).collect(),
			local_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			global_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			dist,
			level_norm_param,
		};
		builder.train(mat);
		builder
	}

}


pub struct HNSWParallelHeapBuilder2<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> {
	_phantom: std::marker::PhantomData<F>,
	n_data: usize,
	n_threads: usize,
	params: HNSWParams<F>,
	n_layers: usize,
	graphs: Vec<HNSWHeapBuildGraph<R,F>>,
	local_layer_ids: Vec<Vec<R>>,
	global_layer_ids: Vec<Vec<R>>,
	dist: Dist,
	level_norm_param: f32,
	rev_edge_cache: Vec<Vec<(usize,R,R,F)>>,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWParallelHeapBuilder2<R, F, Dist> {
	fn initialize_structure(&mut self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
		/* Precompute all point levels */
		let layer_sizes = (0..self.n_layers).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
		let mut point_levels = Vec::with_capacity(self.n_data);
		unsafe{point_levels.set_len(self.n_data)};
		let mut point_level_pos: Vec<usize> = Vec::with_capacity(self.n_data);
		unsafe{point_level_pos.set_len(self.n_data)};
		/* Always insert first point on highest level */
		point_levels[0] = self.n_layers-1;
		layer_sizes[self.n_layers-1].store(1, Relaxed);
		/* Insert all other points */
		let chunk_size = (self.n_data + self.n_threads - 1) / self.n_threads;
		point_levels[1..].chunks_mut(chunk_size).par_bridge().for_each(|point_level_chunk| {
			point_level_chunk.iter_mut().for_each(|point_level_ref| {
				let level = random_level(self.level_norm_param, self.n_layers);
				*point_level_ref = level;
				layer_sizes[level].fetch_add(1, Relaxed);
			})
		});
		/* Remove atomic wrappers */
		let mut layer_sizes = layer_sizes.into_iter().map(|x| x.into_inner()).collect::<Vec<_>>();
		/* Aggregate layer sizes such that the bottom layer has size N */
		(0..self.n_layers-1).rev().for_each(|i| layer_sizes[i] += layer_sizes[i+1]);
		debug_assert!(layer_sizes[0] == self.n_data);
		/* Initialize lowest level */
		let graph0 = &mut self.graphs[0];
		graph0.reserve(self.n_data);
		(0..self.n_data).for_each(|_| graph0.add_node_with_capacity(self.params.lowest_max_degree));
		/* Initialize all other levels */
		unsafe {
			/* Preallocate graphs and ID lookup memory (warning: uninitialized RAM, handle with care!) */
			layer_sizes.iter().skip(1)
			.zip(self.local_layer_ids.iter_mut())
			.zip(self.global_layer_ids.iter_mut())
			.zip(self.graphs.iter_mut().skip(1))
			.par_bridge()
			.for_each(|(((&layer_size, local_ids), global_ids), graph)| {
				local_ids.reserve(layer_size);
				local_ids.set_len(layer_size);
				global_ids.reserve(layer_size);
				global_ids.set_len(layer_size);
				graph.reserve(layer_size);
				(0..layer_size).for_each(|_| graph.add_node_with_capacity(self.params.higher_max_degree));
			});
			/* Set level pos for layer 0 */
			point_level_pos.chunks_mut(chunk_size)
			.zip(point_levels.chunks(chunk_size))
			.enumerate().par_bridge().for_each(|(i_chunk, (point_level_pos_chunk, point_level_chunk))| {
				point_level_pos_chunk.iter_mut().zip(point_level_chunk.iter()).enumerate()
				.filter(|(_,(_,&level))| level==0).for_each(|(i, (point_level_pos_ref, _))| {
					*point_level_pos_ref = i_chunk*chunk_size + i;
				});
			});
			/* Set level pos for all higher levels */
			self.local_layer_ids.iter_mut()
			.zip(self.global_layer_ids.iter_mut())
			.enumerate().par_bridge().for_each(|(i_layer,(local_ids, global_ids))| {
				/* Unsafe reference for (unsafe) parallel write access */
				let unsafe_point_level_pos = std::ptr::addr_of!(point_level_pos) as *mut Vec<usize>;
				/* Incrementing enumerator as the ID arrays start at layer 1 */
				let i_layer = i_layer+1;
				/* Counting local IDs for current and next layer to properly connect them in ID lookups */
				let next_level = i_layer-1;
				let curr_level = i_layer;
				let mut next_layer_offset = R::zero();
				let mut curr_layer_offset = 0;
				/* Setting up local and global ID lookups */
				point_levels.iter().enumerate().for_each(|(i, &level)| {
					if level >= next_level {
						/* Point i requires an entry in the lookups of the current layer */
						if level >= curr_level {
							local_ids[curr_layer_offset] = next_layer_offset;
							global_ids[curr_layer_offset] = R::from_usize(i).unwrap_unchecked();
							if level == curr_level { (&mut *unsafe_point_level_pos)[i] = curr_layer_offset; }
							curr_layer_offset += 1;
						}
						/* Point i requires an entry in the lookups of the next layer */
						next_layer_offset += R::one();
					}
				});
				/* Ensure that all IDs between 0 and N were used for this and next layer */
				debug_assert!(curr_layer_offset == layer_sizes[curr_level]);
				debug_assert!(next_layer_offset.to_usize().unwrap() == layer_sizes[next_level]);
			});
			#[cfg(debug_assertions)]
			{
				point_levels.iter().zip(point_level_pos.iter()).enumerate().for_each(|(i,(&level, &pos))| {
					assert!(layer_sizes[level] > pos);
					let mut pos = pos;
					(1..level+1).rev().for_each(|i_layer| {
						assert_eq!(i, self.global_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked());
						pos = self.local_layer_ids[i_layer-1][pos].to_usize().unwrap_unchecked();
					});
					assert_eq!(i, pos);
				});
			}
		}
		(layer_sizes, point_levels, point_level_pos)
	}
	fn train<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M) {
		/* Do nothing for empty datasets */
		if self.n_data == 0 { return; }
		/* Preallocate graphs and ID lookups and assign levels and level positions for each input */
		let (layer_sizes, point_levels, point_level_pos) = self.initialize_structure();
		/* Burnin single thread for a few samples */
		let n_burnin = self.params.n_parallel_burnin;
		let mut burnin_cache = HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size);
		let mut rev_edges = Vec::new();
		(1..n_burnin+1).for_each(|i| {
			rev_edges.clear();
			self.insert(mat, point_level_pos[i], i, point_levels[i], &mut burnin_cache, &mut rev_edges);
			rev_edges.chunk_by(|(i_layer,_,_,_),(j_layer,_,_,_)| i_layer==j_layer).for_each(|edges_chunk| {
				let i_layer = edges_chunk[0].0;
				let graph = &mut self.graphs[i_layer];
				let max_degree = if i_layer==0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
				edges_chunk.iter().for_each(|&(_,i,j,d)| {
					let adj = graph.view_neighbors_heap_mut(i);
					if adj.iter().all(|&(_,k)| k != j) {
						if adj.size() < max_degree {
							adj.push(d,j);
						} else {
							adj.push_pop(d,j);
						}
					}
				});
			});
		});
		/* Insert edges to the graphs as per HNSW insertion rules */
		let chunk_size = self.n_threads * self.params.insert_minibatch_size;
		let mut thread_caches = (0..self.n_threads).map(|_| HNSWThreadCache::new(&layer_sizes, self.params.max_build_heap_size, self.params.lowest_max_degree, self.params.max_build_frontier_size)).collect::<Vec<_>>();
		let insert_minibatch_size = self.params.insert_minibatch_size;
		let n_data = self.n_data;
		(1+n_burnin..self.n_data).step_by(chunk_size)
		.for_each(|chunk_start| {
			let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
			let chunk_end = (chunk_start+chunk_size).min(n_data);
			unsafe{
				(chunk_start..chunk_end).step_by(insert_minibatch_size)
				.map(|batch_start| batch_start..(batch_start+insert_minibatch_size).min(chunk_end))
				.zip(thread_caches.iter_mut().zip((*unsafe_self_ref).rev_edge_cache.iter_mut()))
				.par_bridge().for_each(|(batch, (thread_cache, rev_edges))| {
					let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
					rev_edges.clear();
					batch.for_each(|i| {
						(*unsafe_self_ref).insert(mat, point_level_pos[i], i, point_levels[i], thread_cache, rev_edges);
					});
				});
			}
			self.add_reverse_edges();
		});
	}
	fn insert<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, mut i: usize, i_global: usize, level: usize, thread_cache: &mut HNSWThreadCache<R,F>, rev_edges: &mut Vec<(usize,R,R,F)>) {
		/* We just assume that the global index 1 is the root entry point */
		let top_entry_dist = self._get_dist(mat, i_global, 0);
		thread_cache.entry_points.clear();
		thread_cache.entry_points.push((top_entry_dist, R::zero()));
		/* Search entry point for the required level */
		(level+1..self.n_layers).rev().for_each(|i_layer| {
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, None);
			let local_ids = &self.local_layer_ids[i_layer-1];
			thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
		});
		(0..level+1).rev().for_each(|i_layer| {
			self._search_layer(mat, i_global, i_layer, unsafe{thread_cache.search_hashsets.get_unchecked_mut(i_layer)}, &mut thread_cache.search_maxheap, &mut thread_cache.frontier_minheap, &mut thread_cache.frontier_dualheap, &mut thread_cache.entry_points, Some(self.params.max_build_heap_size));
			self.insert_layer(mat, i, i_global, i_layer, thread_cache, rev_edges);
			if i_layer > 0 {
				let local_ids = &self.local_layer_ids[i_layer-1];
				thread_cache.entry_points.iter_mut().for_each(|(_,j)| *j = unsafe{local_ids[j.to_usize().unwrap_unchecked()]});
				i = unsafe{local_ids[i].to_usize().unwrap_unchecked()};
			}
		});
	}
	fn insert_layer<M: MatrixDataSource<F>+Sync>(&mut self, mat: &M, i: usize, i_global: usize, layer: usize, thread_cache: &mut HNSWThreadCache<R,F>, rev_edges: &mut Vec<(usize,R,R,F)>) {
		let neighbors = &mut thread_cache.entry_points;
		debug_assert_eq!(neighbors.len(), neighbors.iter().map(|(_,j)| j).collect::<std::collections::HashSet<_>>().len());
		let heuristic_hashset = unsafe{thread_cache.heuristic_hashsets.get_unchecked_mut(layer)};
		/* Translate i into a local (current graph) and a global (dataset/distance computations) ID */
		let i = unsafe{R::from_usize(i).unwrap_unchecked()};
		/* Choose the appropriate max degree */
		let max_neighbors = if layer == 0 { self.params.lowest_max_degree } else { self.params.higher_max_degree };
		/* Unsafe graph reference to have `self` accessible downstream */
		let graph = &mut self.graphs[layer] as *mut HNSWHeapBuildGraph<R,F>;
		unsafe {
			let i_adj = (*graph).view_neighbors_heap_mut(i);
			/* Reduce neighbors to set of max_neighbors points by using a heuristic */
			/* Then bidirectionally link i to all of these neighbors */
			/* Afterwards prune neighborhoods of all affected points back down to max_neighbors */
			if !self.params.insert_heuristic {
				neighbors.iter()
				/* This should almost never matter, but this node can be found by another
					* searching thread while creating the adjacency list. */
				.filter(|&&(_,j)| i != j)
				.take(max_neighbors)
				.for_each(|&(d,j)| i_adj.push(d,j));
			} else {
				/* Maybe extend candidate set with neighbors of neighbors */
				let mut candidates = neighbors.clone();
				if self.params.insert_heuristic_extend {
					heuristic_hashset.clear();
					candidates.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
					(0..neighbors.len()).for_each(|i_neighbor| {
						(*graph).view_neighbors(R::from_usize(i_neighbor).unwrap_unchecked()).iter().for_each(|&(_,j)| {
							if heuristic_hashset.insert(j) {
								let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
								candidates.push((self._get_dist(mat, i_global, j_global.to_usize().unwrap_unchecked()), j));
							}
						});
					});
				}
				/* Make sure that the candidates are truly sorted and unique */
				candidates.sort_unstable_by(|a,b| {
					let dist_cmp = a.0.partial_cmp(&b.0).unwrap();
					if dist_cmp.is_eq() { a.1.cmp(&b.1) } else { dist_cmp }
				});
				remove_duplicates_with_key(&mut candidates, |(_,j)| j);
				/* Prune neighborhoods with relative neighbor heuristic */
				heuristic_hashset.clear();
				heuristic_hashset.insert(i);
				let mut tmp_list = Vec::with_capacity(candidates.len());
				let mut cand_iter = candidates.into_iter().peekable();
				i_adj.iter().for_each(|&(_,j)| _=heuristic_hashset.insert(j));
				while cand_iter.peek().is_some() && heuristic_hashset.contains(&cand_iter.peek().unwrap_unchecked().1) { cand_iter.next(); }
				if cand_iter.peek().is_some() {
					let (dij,j) = cand_iter.next().unwrap_unchecked();
					i_adj.push(dij,j);
					heuristic_hashset.insert(j);
					for (dij,j) in cand_iter {
						if i_adj.size() >= max_neighbors { break; }
						if heuristic_hashset.contains(&j) { continue; }
						if i_adj.iter().all(|&(dik,k)| {
							let j_global = if layer>0 { self.global_layer_ids[layer-1][j.to_usize().unwrap_unchecked()] } else { j };
							let k_global = if layer>0 { self.global_layer_ids[layer-1][k.to_usize().unwrap_unchecked()] } else { k };
							let djk = self._get_dist(mat, j_global.to_usize().unwrap_unchecked(), k_global.to_usize().unwrap_unchecked());
							dik < djk
							// tri_to_cos(dij.to_f32().unwrap_unchecked(),dik.to_f32().unwrap_unchecked(),djk.to_f32().unwrap_unchecked(),true) < 0.5
						}) {
							i_adj.push(dij,j);
						} else {
							tmp_list.push((dij,j));
						}
					};
					if max_neighbors > i_adj.size() {
						/* Add removed edges back in */
						tmp_list.iter().take(max_neighbors-i_adj.size()).for_each(|&x| i_adj.push(x.0,x.1));
					}
				}
			}
			/* Add backward edges */
			i_adj.iter().for_each(|&(d,j)| rev_edges.push((layer,j,i,d)));
		}
	}
	fn add_reverse_edges_single(adj: &mut MaxHeap<F,R>, edges: &mut std::slice::Iter<(usize, R, R, F)>, max_degree: usize) {
		edges.for_each(|&(_,_,j,d)| {
			if adj.iter().all(|&(_,k)| k != j) {
				if adj.size() < max_degree {
					adj.push(d,j);
				} else {
					adj.push_pop(d,j);
				}
			}
		});
	}
	fn add_reverse_edges(&mut self) {
		let n_total_rev_edges = self.rev_edge_cache.iter().map(|x| x.len()).sum();
		let mut total_rev_edges = Vec::with_capacity(n_total_rev_edges);
		self.rev_edge_cache.iter_mut().for_each(|x| total_rev_edges.extend(x.drain(..)));
		total_rev_edges.par_sort_unstable_by(|(i_layer,i_u, _, _), (j_layer,j_u, _, _)| {
			let layer_cmp = i_layer.cmp(j_layer);
			if layer_cmp.is_eq() { i_u.cmp(j_u) } else { layer_cmp }
		});
		total_rev_edges.par_chunk_by(|(i_layer, i_u, _, _), (j_layer,j_u, _, _)| i_u==j_u && i_layer==j_layer).for_each(|edges_chunk| {
			let unsafe_self_ref = std::ptr::addr_of!(*self) as *mut Self;
			unsafe {
				let (i_layer, i, _, _) = edges_chunk[0];
				let max_degree = if i_layer==0 { (*unsafe_self_ref).params.lowest_max_degree } else { (*unsafe_self_ref).params.higher_max_degree };
				let adj = (&mut (*unsafe_self_ref).graphs)[i_layer].view_neighbors_heap_mut(i);
				Self::add_reverse_edges_single(adj, &mut edges_chunk.iter(), max_degree);
			}
		});
	}
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync+Send> HNSWStyleBuilder<R, F, Dist> for HNSWParallelHeapBuilder2<R, F, Dist> {
	type Params = HNSWParams<F>;
	type Graph = HNSWHeapBuildGraph<R,F>;
	#[inline(always)]
	fn _mut_graphs(&mut self) -> &mut Vec<HNSWHeapBuildGraph<R,F>> { &mut self.graphs }
	#[inline(always)]
	fn _graphs(&self) -> &Vec<HNSWHeapBuildGraph<R,F>> { &self.graphs }
	#[inline(always)]
	fn _global_layer_ids(&self) -> &Vec<Vec<R>> { &self.global_layer_ids }
	#[inline(always)]
	fn _dist(&self) -> &Dist { &self.dist }
	#[inline(always)]
	fn _max_build_heap_size(&self) -> usize { self.params.max_build_heap_size }
	#[inline(always)]
	fn _max_build_frontier_size(&self) -> Option<usize> { self.params.max_build_frontier_size }
	#[inline(always)]
	fn _into_parts(self) -> (Vec<HNSWHeapBuildGraph<R,F>>, Vec<Vec<R>>, Vec<Vec<R>>, Dist) {
		(self.graphs, self.local_layer_ids, self.global_layer_ids, self.dist)
	}
	#[inline(always)]
	fn _max_degrees(&self) -> (usize, usize) {
		(self.params.lowest_max_degree, self.params.higher_max_degree)
	}
	fn base_init<M: MatrixDataSource<F>+Sync>(mat: &M, dist: Dist, params: Self::Params) -> Self {
		let n_data = mat.n_rows();
		assert!(n_data < R::max_value().to_usize().unwrap());
		let n_threads = current_num_threads();
		let level_norm_param = params.level_norm_param_override.unwrap_or(1.0 / (params.higher_max_degree as f32).ln());
		/* By the law of large numbers, this is the appropriate number of layers */
		let n_layers = (((n_data as f32).ln() * level_norm_param).floor() as usize + 1).min(params.max_layers);
		let rev_edge_cache = vec![Vec::with_capacity(params.insert_minibatch_size * params.lowest_max_degree * n_layers); n_threads];
		let mut builder = Self {
			_phantom: std::marker::PhantomData,
			n_data,
			n_threads,
			params,
			n_layers,
			graphs: (0..n_layers).map(|_| HNSWHeapBuildGraph::new()).collect(),
			local_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			global_layer_ids: (0..n_layers-1).map(|_| Vec::new()).collect(),
			dist,
			level_norm_param,
			rev_edge_cache,
		};
		builder.train(mat);
		builder
	}

}



pub fn load_hnswlib_fat_parts<F: SyncFloat>(file: &str) -> (Array2<F>, usize, usize, usize, usize, Vec<Vec<Vec<usize>>>, Vec<usize>, usize) {
	use std::fs::File;
	use std::io::BufReader;
	use std::io::Read;
	let mut reader = BufReader::new(File::open(file).unwrap());
	/* Load the entire file into RAM */
	let mut file_data = Vec::new();
	reader.read_to_end(&mut file_data).unwrap();
	fn get_type_at<T: Sized+Copy>(data: &Vec<u8>, offset: usize) -> T {
		unsafe{*(data[offset..offset+std::mem::size_of::<T>()].as_ptr() as *const T)}
	}
	fn get_vec<T: Sized+Copy>(data: &Vec<u8>, offset: usize, count: usize) -> Vec<T> {
		let t_size = std::mem::size_of::<T>();
		(0..count).map(|i| get_type_at(data, offset + i*t_size)).collect()
	}
	let curr_elements = get_type_at::<u64>(&file_data, 16);
	let size_per_elem = get_type_at::<u64>(&file_data, 24);
	let max_level = get_type_at::<i32>(&file_data, 48) as usize;
	let entry_point = get_type_at::<u32>(&file_data, 52) as usize;
	let max_m = get_type_at::<u64>(&file_data, 56);
	let max_m0 = get_type_at::<u64>(&file_data, 64);
	let data_level_0_mem_size = curr_elements * size_per_elem;
	let size_links_level0 = max_m0 * 4 + 4;
	let dim = (data_level_0_mem_size / curr_elements - 8 - size_links_level0) / 4;
	let mut data = Array2::from_elem((curr_elements as usize, dim as usize), F::zero());
	/* Read global IDs from level 0 data and compute the order of storage */
	let mut ids = Vec::with_capacity(curr_elements as usize);
	let level_0_start = 12*8 as usize;
	let level_0_end = level_0_start + data_level_0_mem_size as usize;
	(0..curr_elements as usize).for_each(|i| {
		let offset = level_0_start + (i+1)*(size_per_elem as usize) - 8;
		let id = get_type_at::<u64>(&file_data, offset);
		ids.push(id as usize);
		get_vec::<f32>(
			&file_data,
			level_0_start
			+ i*size_per_elem as usize
			+ 4
			+ 4*max_m0 as usize,
			dim as usize
		).iter().enumerate().for_each(|(j, &x)| {
			data[[i,j]] = F::from_f32(x).unwrap();
		});
	});
	let mut order = (0..curr_elements as usize).collect::<Vec<_>>();
	order.sort_by_key(|&i| ids[i]);
	assert!(order.iter().enumerate().all(|(i,&j)| i==ids[j] as usize), "Global IDs are not contiguous, unique, or starting at 0");
	/* Iterate over the data and build the adjacency lists */
	let mut adjs: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); curr_elements as usize]; max_level+1];
	let mut higher_offset = level_0_end;
	let higher_adj_size = 4*max_m as usize + 4;
	let mut node_levels: Vec<usize> = vec![0; curr_elements as usize];
	(0..curr_elements as usize).for_each(|i_obj| {
		let n_neighbors_0 = get_type_at::<u32>(&file_data, level_0_start + i_obj*size_per_elem as usize) as usize;
		assert!(n_neighbors_0 <= max_m0 as usize, "Too many neighbors in level 0: {:?}", n_neighbors_0);
		let adj_0 = get_vec::<u32>(&file_data, level_0_start + i_obj*size_per_elem as usize + 4, n_neighbors_0);
		adjs[0][ids[i_obj]].reserve(n_neighbors_0);
		adj_0.into_iter().for_each(|x| adjs[0][ids[i_obj]].push(ids[x as usize]));
		assert!(adjs[0][ids[i_obj]].len() == n_neighbors_0, "Adjacency list length mismatch at layer 0: {:?} != {:?}", adjs[0][ids[i_obj]].len(), n_neighbors_0);
		let n_bytes_higher_adjs: usize = get_type_at::<u32>(&file_data, higher_offset) as usize;
		assert!(n_bytes_higher_adjs % higher_adj_size == 0, "Invalid higher adjacency list size: {:?}", n_bytes_higher_adjs);
		higher_offset += 4;
		let level = n_bytes_higher_adjs / higher_adj_size;
		node_levels[ids[i_obj]] = level;
		(0..level).for_each(|i_level| {
			let list_pos = higher_offset + i_level*higher_adj_size;
			let n_neighbors_i = get_type_at::<u32>(&file_data, list_pos) as usize;
			assert!(n_neighbors_i <= max_m as usize, "Too many neighbors in level {:?}: {:?}", i_level+1, n_neighbors_i);
			let adj_i = get_vec::<u32>(&file_data, list_pos + 4, n_neighbors_i);
			adjs[i_level+1][ids[i_obj]].reserve(n_neighbors_i);
			adj_i.into_iter().for_each(|x| adjs[i_level+1][ids[i_obj]].push(ids[x as usize]));
			assert!(adjs[i_level+1][ids[i_obj]].len() == n_neighbors_i, "Adjacency list length mismatch at layer {:?}: {:?} != {:?}", i_level+1, adjs[i_level+1][ids[i_obj]].len(), n_neighbors_i);
		});
		higher_offset += n_bytes_higher_adjs;
	});
	(data, max_m as usize, max_m0 as usize, max_level, curr_elements as usize, adjs, node_levels, entry_point)
}
pub fn load_hnswlib_parts<R: SyncUnsignedInteger, F: SyncFloat>(file: &str) -> (Array2<F>, Vec<DirLoLGraph<R>>, Vec<Vec<R>>, Vec<Vec<R>>, R) {
	let (data, _, _, max_level, curr_elements, adjs, node_levels, entry_point) = load_hnswlib_fat_parts(file);
	/* Translate adjacency lists into graphs and ID maps */
	let mut graphs: Vec<DirLoLGraph<R>> = (0..max_level+1).map(|_| DirLoLGraph::new()).collect();
	let mut local_id_maps = vec![Vec::new(); max_level];
	let mut global_id_maps = vec![Vec::new(); max_level];
	adjs.iter().enumerate().for_each(|(i_level, adj)| {
		let mut adj_id_map = Vec::with_capacity(curr_elements);
		let mut found_nodes = 0;
		adj.iter().enumerate().for_each(|(i_node, i_adj)| {
			if node_levels[i_node] >= i_level {
				adj_id_map.push(R::from_usize(found_nodes).unwrap());
				found_nodes += 1;
				graphs[i_level].add_node_with_capacity(i_adj.len());
				if i_level == 1 {
					local_id_maps[0].push(R::from_usize(i_node).unwrap());
					global_id_maps[0].push(R::from_usize(i_node).unwrap());
				} else if i_level > 1 {
					let next_layer_id = local_id_maps[i_level-2].len()-1;
					local_id_maps[i_level-1].push(R::from_usize(next_layer_id).unwrap());
					global_id_maps[i_level-1].push(R::from_usize(i_node).unwrap());
				}
			} else {
				adj_id_map.push(R::max_value());
			}
		});
		adj.iter().enumerate().for_each(|(i_node, i_adj)| {
			i_adj.iter().for_each(|&j_node| {
				assert!(adj_id_map[i_node] < R::max_value());
				assert!(adj_id_map[j_node] < R::max_value());
				graphs[i_level].add_edge(adj_id_map[i_node], adj_id_map[j_node]);
			})
		});
	});
	/* Ensure that the highest level graph is actually occupied */
	while graphs[graphs.len()-1].n_vertices() == 0 {
		graphs.pop();
		local_id_maps.pop();
		global_id_maps.pop();
	}
	/* Translate entry point to local ID in top level graph */
	let entry_point = R::from(entry_point).unwrap();
	let entry_point = global_id_maps[global_id_maps.len()-1].iter().position(|&x| x == entry_point).unwrap();
	let entry_point = R::from(entry_point).unwrap();
	/* Return the loaded values */
	(data, graphs, local_id_maps, global_id_maps, entry_point)
}
pub fn load_hnswlib_fat<R: SyncUnsignedInteger, F: SyncFloat>(file: &str) -> GreedyLayeredGraphIndex<R, F, SquaredEuclideanDistance<F>, Array2<F>, FatDirGraph<R>>{
	let (data, max_m, max_m0, _, curr_elements, adjs, _, entry_point) = load_hnswlib_fat_parts(file);
	/* Translate adjacency lists into graphs */
	let graphs = adjs.into_iter().enumerate().map(|(i_graph, adj)| {
		let mut graph = FatDirGraph::new(if i_graph==0 {max_m0} else {max_m});
		graph.reserve(curr_elements);
		(0..curr_elements).for_each(|_| graph.add_node());
		adj.into_iter().enumerate().for_each(|(i_node, neighbors)| {
			neighbors.into_iter().for_each(|j_node| {
				graph.add_edge(R::from_usize(i_node).unwrap(), R::from_usize(j_node).unwrap());
			})
		});
		graph
	}).collect();
	GreedyLayeredGraphIndex::new(
		data,
		graphs,
		Vec::new(),
		Vec::new(),
		SquaredEuclideanDistance::new(),
		1,
		Some(vec![R::from(entry_point).unwrap()])
	)
}
pub fn load_hnswlib_fat_capped<R: SyncUnsignedInteger, F: SyncFloat>(file: &str, max_frontier_size: usize) -> GreedyCappedLayeredGraphIndex<R, F, SquaredEuclideanDistance<F>, Array2<F>, FatDirGraph<R>>{
	load_hnswlib_fat(file).into_capped(max_frontier_size)
}
pub fn load_hnswlib<R: SyncUnsignedInteger, F: SyncFloat>(file: &str) -> GreedyLayeredGraphIndex<R, F, SquaredEuclideanDistance<F>, Array2<F>, DirLoLGraph<R>>{
	let (data, graphs, local_id_maps, global_id_maps, entry_point) = load_hnswlib_parts(file);
	GreedyLayeredGraphIndex::new(
		data,
		graphs,
		local_id_maps,
		global_id_maps,
		SquaredEuclideanDistance::new(),
		1,
		Some(vec![entry_point])
	)
}
pub fn load_hnswlib_capped<R: SyncUnsignedInteger, F: SyncFloat>(file: &str, max_frontier_size: usize) -> GreedyCappedLayeredGraphIndex<R, F, SquaredEuclideanDistance<F>, Array2<F>, DirLoLGraph<R>>{
	let (data, graphs, local_id_maps, global_id_maps, entry_point) = load_hnswlib_parts(file);
	GreedyCappedLayeredGraphIndex::new(
		data,
		graphs,
		local_id_maps,
		global_id_maps,
		SquaredEuclideanDistance::new(),
		1,
		max_frontier_size,
		Some(vec![entry_point]),
	)
}





#[cfg(test)]
mod tests {
	use crate::hnsw::*;
	// use crate::hnsw::single_threaded::*;
	use ndarray::{Array2,Slice};
	use ndarray_rand::rand_distr::Normal;
	use rand::prelude::Distribution;
	use ndarray::Axis;
	use std::time::Instant;
	use rand::distributions::Uniform;

	#[test]
	fn random_level_test() {
		let n_values = 1_000_000;
		let higher_max_degree = 8;
		let level_norm_param = 1.0 / (higher_max_degree as f32).ln();
		let n_layers = (((n_values as f32).ln() / (higher_max_degree as f32).ln()).floor().max(2.0) as usize) - 1;
		let mut layer_cnts = vec![0; n_layers];
		(0..n_values).for_each(|_| {
			let layer = random_level(level_norm_param, n_layers);
			layer_cnts[layer] += 1;
		});
		(0..n_layers-1).for_each(|i| {
			assert!(
				((layer_cnts[i] as f32 / layer_cnts[i+1] as f32)-higher_max_degree as f32).abs() < 1.5,
				"Too many/few points in layer {:?} compared to layer {:?}.\nQuotient is {:?} but should be close to {:?}.\n{:?}",
				i,i+1,(layer_cnts[i] as f32 / layer_cnts[i+1] as f32),higher_max_degree,layer_cnts
			);
		});
	}

	#[test]
	fn rounds_hnsw_construction() {
		let (n,d) = (10_000, 20);
		let rng = Normal::new(0.0, 1.0).unwrap();
		let data: Array2<f64> = Array2::from_shape_fn((n, d), |_| rng.sample(&mut rand::thread_rng()));
		let params = HNSWParams::new()
		.with_insert_heuristic(false)
		.with_insert_heuristic_extend(false)
		.with_n_rounds(3);
		let graph_time = std::time::Instant::now();
		println!("Entering graph building.");
		let _graph = HNSWParallelHeapBuilder::<u64,_,_>::build(data, SquaredEuclideanDistance::new(), params, 1);
		println!("Graph construction: {:.2?}", graph_time.elapsed());
	}
	
	#[test]
	fn flooding_hnsw_construction() {
		let (n,d) = (10_000, 20);
		let rng = Normal::new(0.0, 1.0).unwrap();
		let data: Array2<f64> = Array2::from_shape_fn((n, d), |_| rng.sample(&mut rand::thread_rng()));
		let params = HNSWParams::new()
		.with_insert_heuristic_extend(false);
		let graph_time = std::time::Instant::now();
		let _graph = FloodingHNSWBuilder::<u64,_,_>::build(data, EuclideanDistance::new(), params, 1);
		println!("Graph construction: {:.2?}", graph_time.elapsed());
	}
	
	#[test]
	fn hnsw_query() {
		/* Limit global thread pool size */
		// rayon::ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();
		/* Parameters */
		let (nd, nq, d, k) = (1_000_000, 1000, 50, 10);
		let euc = SquaredEuclideanDistance::new();
		/* Data initialization */
		let init_time = Instant::now();
		type R = usize;
		type F = f32;
		type Dist = SquaredEuclideanDistance<F>;
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
		let dist = Dist::new();
		let params = HNSWParams::new()
		.with_n_parallel_burnin(1000)
		.with_insert_heuristic(true)
		.with_insert_heuristic_extend(false)
		.with_higher_max_degree(254)
		.with_lowest_max_degree(254)
		// .with_max_build_heap_size(100)
		// .with_max_build_heap_size(10*d)
		.with_max_build_heap_size(2*254)
		// .with_max_build_frontier_size(None)
		// .with_max_build_frontier_size(Some(20))
		.with_max_build_frontier_size(Some(50))
		.with_level_norm_param_override(Some(0.5))
		.with_insert_minibatch_size(1000)
		.with_max_layers(16)
		// .with_max_layers(1)
		.with_n_rounds(3)
		;
		type BuilderType = HNSWParallelHeapBuilder<R,F,Dist>;
		// type BuilderType = HNSWParallelSENHeapBuilder<R,F,Dist>;
		// type BuilderType = HNSWParallelHeapBuilder2<R,F,Dist>;
		// type BuilderType = HNSWParallelPresortedHeapBuilder<R,F,Dist>;
		let index1 = BuilderType::build(data.view(), dist, params, 1);
		println!("Graph construction ({:?}): {:.2?}", std::any::type_name::<BuilderType>(), build_time.elapsed());
		/* Verify graphs */
		(0..index1.layer_count()).for_each(|i_layer| {
			let graph = &index1.graphs()[i_layer];
			/* Verify no duplicate edges, loops, or "escaping" edges */
			(0..graph.n_vertices()).for_each(|i| {
				let neighbors = graph.view_neighbors(i);
				assert!(neighbors.len() <= if i_layer==0 {params.lowest_max_degree} else {params.higher_max_degree}, "Too many neighbors in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors.len());
				assert_eq!(neighbors.iter().collect::<std::collections::HashSet<_>>().len(), neighbors.len(), "Duplicate neighbors in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors);
				assert!(neighbors.iter().all(|&j| j!=i), "Loop in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors);
				assert!(neighbors.iter().all(|&j| j < graph.n_vertices()), "Escaping edge in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors);
			});
		});
		println!("Graph sizes: {:.2?}", (0..index1.layer_count()).map(|i_layer| {
			let graph = &index1.graphs()[i_layer];
			graph.n_vertices()
		}).collect::<Vec<_>>());
		println!("Average out degrees: {:.2?}", (0..index1.layer_count()).map(|i_layer| {
			let graph = &index1.graphs()[i_layer];
			graph.n_edges() as f32 / (graph.n_vertices() as f32)
		}).collect::<Vec<_>>());
		let index2 = GreedyCappedLayeredGraphIndex::new(
			data.view(),
			index1.graphs().iter().map(|g|g.as_dir_lol_graph()).collect(),
			(1..index1.graphs().len()).map(|i| index1.get_local_layer_ids(i).unwrap().clone()).collect(),
			(1..index1.graphs().len()).map(|i| index1.get_global_layer_ids(i).unwrap().clone()).collect(),
			SquaredEuclideanDistance::new(),
			index1.higher_level_max_heap_size(),
			3*k,
			None,
		);
		/* Brute force queries */
		let bruteforce_time = Instant::now();
		let (bruteforce_ids, _) = bruteforce_neighbors(&data, &queries, &euc, k);
		println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());
		/* HNSW queries */
		let hnsw_time = Instant::now();
		#[allow(unused)]
		let (hnsw_ids1, hnsw_dists1) = index1.greedy_search_batch(&queries, k, k);
		println!("HNSW queries 1: {:.2?}", hnsw_time.elapsed());
		let hnsw_time = Instant::now();
		#[allow(unused)]
		let (hnsw_ids2, hnsw_dists2) = index2.greedy_search_batch(&queries, k, k);
		println!("HNSW queries 2: {:.2?}", hnsw_time.elapsed());
		/* Compute and print recall */
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(hnsw_ids1.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let hnsw_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&hnsw_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 1: {:.2}%", recall*100f32);
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(hnsw_ids2.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let hnsw_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&hnsw_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 2: {:.2}%", recall*100f32);
	}


	#[test]
	fn hnsw_probs_test() {
		/* Limit global thread pool size */
		// rayon::ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();
		/* Parameters */
		let (nd, nq, d, k, search_heap_size) = (100_000, 1000, 50, 1, 10);
		let euc = SquaredEuclideanDistance::new();
		/* Data initialization */
		let init_time = Instant::now();
		type R = usize;
		type F = f32;
		type Dist = SquaredEuclideanDistance<F>;
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
		let dist = Dist::new();
		let params = HNSWParams::new()
			.with_n_parallel_burnin(1000)
			.with_insert_heuristic(false)
			.with_insert_heuristic_extend(false)
			.with_higher_max_degree(20)
			.with_lowest_max_degree(20)
			.with_max_build_heap_size(500)
			.with_max_build_frontier_size(None)
			.with_level_norm_param_override(Some(0.5))
			.with_insert_minibatch_size(1000)
			.with_max_layers(1)
			.with_n_rounds(3)
		;
		type BuilderType = HNSWParallelHeapBuilder<R,F,Dist>;
		let index1 = BuilderType::build(data.view(), dist, params, 1);
		println!("Graph construction ({:?}): {:.2?}", std::any::type_name::<BuilderType>(), build_time.elapsed());
		if 1>0 { /* Verify graphs */
			(0..index1.layer_count()).for_each(|i_layer| {
				let graph = &index1.graphs()[i_layer];
				/* Verify no duplicate edges, loops, or "escaping" edges */
				(0..graph.n_vertices()).for_each(|i| {
					let neighbors = graph.view_neighbors(i);
					assert!(neighbors.len() <= if i_layer==0 {params.lowest_max_degree} else {params.higher_max_degree}, "Too many neighbors in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors.len());
					assert_eq!(neighbors.iter().collect::<std::collections::HashSet<_>>().len(), neighbors.len(), "Duplicate neighbors in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors);
					assert!(neighbors.iter().all(|&j| j!=i), "Loop in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors);
					assert!(neighbors.iter().all(|&j| j < graph.n_vertices()), "Escaping edge in graph {:?} at vertex {:?}: {:?}", i_layer, i, neighbors);
				});
			});
			println!("Graph sizes: {:.2?}", (0..index1.layer_count()).map(|i_layer| {
				let graph = &index1.graphs()[i_layer];
				graph.n_vertices()
			}).collect::<Vec<_>>());
			println!("Average out degrees: {:.2?}", (0..index1.layer_count()).map(|i_layer| {
				let graph = &index1.graphs()[i_layer];
				graph.n_edges() as f32 / (graph.n_vertices() as f32)
			}).collect::<Vec<_>>());
		}
		let index2 = GreedyCappedLayeredGraphIndex::new(
			data.view(),
			index1.graphs().iter().map(|g|g.as_dir_lol_graph()).collect(),
			(1..index1.graphs().len()).map(|i| index1.get_local_layer_ids(i).unwrap().clone()).collect(),
			(1..index1.graphs().len()).map(|i| index1.get_global_layer_ids(i).unwrap().clone()).collect(),
			SquaredEuclideanDistance::new(),
			index1.higher_level_max_heap_size(),
			3*k,
			None,
		);
		/* Brute force queries */
		let bruteforce_time = Instant::now();
		let (bruteforce_ids, _) = bruteforce_neighbors(&data, &queries, &euc, k);
		println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());
		/* HNSW queries */
		let hnsw_time = Instant::now();
		#[allow(unused)]
		let (hnsw_ids1, hnsw_dists1) = index1.greedy_search_batch(&queries, k, search_heap_size);
		println!("HNSW queries 1: {:.2?}", hnsw_time.elapsed());
		let hnsw_time = Instant::now();
		#[allow(unused)]
		let (hnsw_ids2, hnsw_dists2) = index2.greedy_search_batch(&queries, k, search_heap_size);
		println!("HNSW queries 2: {:.2?}", hnsw_time.elapsed());
		/* Compute and print recall */
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(hnsw_ids1.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let hnsw_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&hnsw_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 1: {:.2}%", recall*100f32);
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(hnsw_ids2.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let hnsw_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&hnsw_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 2: {:.2}%", recall*100f32);
	}
}
