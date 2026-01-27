

// struct PyArrayDataSource

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use graphidx::{graphs::{DirLoLGraph, FatDirGraph, Graph}, indices::{GraphIndex, GreedyCappedLayeredGraphIndex, GreedyCappedSingleGraphIndex, GreedyLayeredGraphIndex, GreedySingleGraphIndex}, measures::SquaredEuclideanDistance, types::UnsignedInteger, graph_ops::{extend_random_edges, fill_random_edges}};
use pyo3::prelude::*;

use crate::{hnsw::{FloodingHNSWBuilder, FloodingHNSWSENBuilder, HNSWParallelHeapBuilder, HNSWParallelSENHeapBuilder, HNSWParams, HNSWSENParams, HNSWStyleBuilder}, rnn::{RNNStyleBuilder, SENParams}};

/* Conversion code to handle different ndarray versions in this crate and numpy dependencies */
fn arr1_rust_to_py<T>(arr: Array1<T>) -> numpy::ndarray::Array1<T> {
	numpy::ndarray::Array1::from_vec(arr.into_raw_vec())
}
fn arr2_rust_to_py<T>(arr: Array2<T>) -> numpy::ndarray::Array2<T> {
	let shape = arr.shape();
	unsafe{numpy::ndarray::Array2::from_shape_vec_unchecked((shape[0],shape[1]), arr.into_raw_vec())}
}
fn arrview1_py_to_rust<T>(arr: numpy::ndarray::ArrayView1<T>) -> ArrayView1<'static, T> {
	let shape = arr.shape();
	unsafe{ArrayView1::from_shape_ptr((shape[0],), arr.as_ptr() as *const T)}
}
fn arrview2_py_to_rust<T>(arr: numpy::ndarray::ArrayView2<T>) -> ArrayView2<'static, T> {
	let shape = arr.shape();
	unsafe{ArrayView2::from_shape_ptr((shape[0],shape[1]), arr.as_ptr() as *const T)}
}

#[pyclass]
pub struct GraphStats {
	#[pyo3(get)]
	pub n_nodes: usize,
	#[pyo3(get)]
	pub n_edges: usize,
	#[pyo3(get)]
	pub max_degree: usize,
	#[pyo3(get)]
	pub min_degree: usize,
	#[pyo3(get)]
	pub avg_degree: f64,
	#[pyo3(get)]
	pub std_degree: f64,
}
impl GraphStats {
	pub fn from_graph<R: UnsignedInteger, G: Graph<R>>(g: &G) -> Self {
		let n_nodes = g.n_vertices();
		let (mut n_edges, mut max_degree, mut min_degree, mut ssq_degree) = (0, 0, usize::MAX, 0);
		(0..n_nodes).for_each(|i| {
			let degree = g.degree(R::from(i).unwrap());
			n_edges += degree;
			max_degree = max_degree.max(degree);
			min_degree = min_degree.min(degree);
			ssq_degree += degree*degree;
		});
		let mssq_degree = ssq_degree as f64 / n_nodes as f64;
		let avg_degree = n_edges as f64 / n_nodes as f64;
		let var_degree = mssq_degree - avg_degree*avg_degree;
		let std_degree = var_degree.sqrt();
		Self {
			n_nodes: n_nodes,
			n_edges: n_edges,
			max_degree: max_degree,
			min_degree: min_degree,
			avg_degree: avg_degree,
			std_degree: std_degree,
		}
	}
}


type GSIndex<M> = GreedySingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
type GCSIndex<M> = GreedyCappedSingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
// type FGSIndex<M> = GreedySingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;
// type FGCSIndex<M> = GreedyCappedSingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;
type GLIndex<M> = GreedyLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
type GCLIndex<M> = GreedyCappedLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, DirLoLGraph<usize>>;
type FGLIndex<M> = GreedyLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;
type FGCLIndex<M> = GreedyCappedLayeredGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, M, FatDirGraph<usize>>;


pub enum IndexOneOf<A: GraphIndex<usize,f32,SquaredEuclideanDistance<f32>>, B: GraphIndex<usize,f32,SquaredEuclideanDistance<f32>>> {
	A(A),
	B(B),
	None,
}
#[allow(dead_code)]
impl<A: GraphIndex<usize,f32,SquaredEuclideanDistance<f32>>, B: GraphIndex<usize,f32,SquaredEuclideanDistance<f32>>> IndexOneOf<A,B> {
	fn greedy_search(&self, queries: &ArrayView1<f32>, k: usize, max_heap_size: usize) -> (Array1<usize>, Array1<f32>) {
		match self {
				IndexOneOf::A(a) => a.greedy_search(queries, k, max_heap_size, &mut a._new_search_cache(max_heap_size)),
				IndexOneOf::B(b) => b.greedy_search(queries, k, max_heap_size, &mut b._new_search_cache(max_heap_size)),
				IndexOneOf::None => panic!(),
		}
	}
	fn greedy_search_batch(&self, queries: &ArrayView2<f32>, k: usize, max_heap_size: usize) -> (Array2<usize>, Array2<f32>) {
		match self {
				IndexOneOf::A(a) => a.greedy_search_batch(queries, k, max_heap_size),
				IndexOneOf::B(b) => b.greedy_search_batch(queries, k, max_heap_size),
				IndexOneOf::None => panic!(),
		}
	}
	fn as_a(&self) -> Option<&A> {
		match self {
			IndexOneOf::A(a) => Some(a),
			IndexOneOf::B(_) => None,
			IndexOneOf::None => None,
		}
	}
	fn as_b(&self) -> Option<&B> {
		match self {
			IndexOneOf::A(_) => None,
			IndexOneOf::B(b) => Some(b),
			IndexOneOf::None => None,
		}
	}
	fn as_a_mut(&mut self) -> Option<&mut A> {
		match self {
			IndexOneOf::A(a) => Some(a),
			IndexOneOf::B(_) => None,
			IndexOneOf::None => None,
		}
	}
	fn as_b_mut(&mut self) -> Option<&mut B> {
		match self {
			IndexOneOf::A(_) => None,
			IndexOneOf::B(b) => Some(b),
			IndexOneOf::None => None,
		}
	}
	fn is_a(&self) -> bool {
		match self {
			IndexOneOf::A(_) => true,
			IndexOneOf::B(_) => false,
			IndexOneOf::None => false,
		}
	}
	fn is_b(&self) -> bool {
		match self {
			IndexOneOf::A(_) => false,
			IndexOneOf::B(_) => true,
			IndexOneOf::None => false,
		}
	}
	fn into_a<F: FnOnce(B) -> A>(self, fun: F) -> Self {
		match self {
			IndexOneOf::A(_) => self,
			IndexOneOf::B(b) => IndexOneOf::A(fun(b)),
			IndexOneOf::None => self,
		}
	}
	fn into_b<F: FnOnce(A) -> B>(self, fun: F) -> Self {
		match self {
			IndexOneOf::A(a) => IndexOneOf::B(fun(a)),
			IndexOneOf::B(_) => self,
			IndexOneOf::None => self,
		}
	}
}


macro_rules! generic_graph_index_funs {
	($type: ident) => {
		#[pymethods]
		impl $type {
			#[pyo3(signature = (query, k, max_heap_size=None))]
			fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
				unsafe {
					let (ids, dists) = self.index.greedy_search(
						&arrview1_py_to_rust(query.as_array()),
						k,
						max_heap_size.unwrap_or(2*k),
					);
					(
						PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
						PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
					)
				}
			}
			#[pyo3(signature = (queries, k, max_heap_size=None))]
			fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
				unsafe {
					let (ids, dists) = self.index.greedy_search_batch(
						&arrview2_py_to_rust(queries.as_array()),
						k,
						max_heap_size.unwrap_or(2*k),
					);
					(
						PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
						PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
					)
				}
			}
			#[getter]
			fn get_max_frontier_size(&self) -> Option<usize> {
				self.max_frontier_size
			}
			#[setter]
			fn set_max_frontier_size(&mut self, max_frontier_size: Option<usize>) {
				self.max_frontier_size = max_frontier_size;
				if self.max_frontier_size.is_none() {
					if self.index.is_b() {
						let mut index = IndexOneOf::None;
						std::mem::swap(&mut self.index, &mut index);
						let mut index = index.into_a(|index| index.into_uncapped());
						std::mem::swap(&mut self.index, &mut index);
					}
				} else {
					if self.index.is_a() {
						let mut index = IndexOneOf::None;
						std::mem::swap(&mut self.index, &mut index);
						let mut index = index.into_b(|index| index.into_capped(max_frontier_size.unwrap()));
						std::mem::swap(&mut self.index, &mut index);
					} else {
						self.index.as_b_mut().unwrap().set_max_frontier_size(max_frontier_size.unwrap());
					}
				}
			}
		}
	};
	(layered $type: ident) => {
		generic_graph_index_funs!($type);
		#[pymethods]
		impl $type {
			fn get_graph_stats(&self) -> Vec<GraphStats> {
				match &self.index {
					IndexOneOf::A(index) => index.graphs().iter().map(|g| GraphStats::from_graph(g)).collect(),
					IndexOneOf::B(index) => index.graphs().iter().map(|g| GraphStats::from_graph(g)).collect(),
					IndexOneOf::None => panic!(),
				}
			}
			fn get_neighbors(&self, layer: usize, node: usize) -> Vec<usize> {
				match &self.index {
					IndexOneOf::A(index) => index.graphs()[layer].neighbors(node),
					IndexOneOf::B(index) => index.graphs()[layer].neighbors(node),
					IndexOneOf::None => panic!(),
				}
			}
			fn get_next_layer_id(&self, layer: usize, node: usize) -> usize {
				if layer == 0 { return node; }
				match &self.index {
					IndexOneOf::A(index) => index.get_local_layer_ids(layer).unwrap()[node],
					IndexOneOf::B(index) => index.get_local_layer_ids(layer).unwrap()[node],
					IndexOneOf::None => panic!(),
				}
			}
			fn get_global_id(&self, layer: usize, node: usize) -> usize {
				if layer == 0 { return node; }
				match &self.index {
					IndexOneOf::A(index) => index.get_global_layer_ids(layer).unwrap()[node],
					IndexOneOf::B(index) => index.get_global_layer_ids(layer).unwrap()[node],
					IndexOneOf::None => panic!(),
				}
			}
			fn extend_random_edges(&mut self, layer: usize, num_edges: usize) {
				match &mut self.index {
					IndexOneOf::A(index) => extend_random_edges(index.graphs_mut().get_mut(layer).unwrap(), num_edges),
					IndexOneOf::B(index) => extend_random_edges(index.graphs_mut().get_mut(layer).unwrap(), num_edges),
					IndexOneOf::None => panic!(),
				}
			}
			fn fill_random_edges(&mut self, layer: usize, num_edges: usize) {
				match &mut self.index {
					IndexOneOf::A(index) => fill_random_edges(index.graphs_mut().get_mut(layer).unwrap(), num_edges),
					IndexOneOf::B(index) => fill_random_edges(index.graphs_mut().get_mut(layer).unwrap(), num_edges),
					IndexOneOf::None => panic!(),
				}
			}
		}
	};
	(single $type: ident) => {
		generic_graph_index_funs!($type);
		#[pymethods]
		impl $type {
			fn get_graph_stats(&self) -> GraphStats {
				match &self.index {
					IndexOneOf::A(index) => GraphStats::from_graph(index.graph()),
					IndexOneOf::B(index) => GraphStats::from_graph(index.graph()),
					IndexOneOf::None => panic!(),
				}
			}
			fn get_neighbors(&self, node: usize) -> Vec<usize> {
				match &self.index {
					IndexOneOf::A(index) => index.graph().neighbors(node),
					IndexOneOf::B(index) => index.graph().neighbors(node),
					IndexOneOf::None => panic!(),
				}
			}
			fn extend_random_edges(&mut self, num_edges: usize) {
				match &mut self.index {
					IndexOneOf::A(index) => extend_random_edges(index.graph_mut(), num_edges),
					IndexOneOf::B(index) => extend_random_edges(index.graph_mut(), num_edges),
					IndexOneOf::None => panic!(),
				}
			}
			fn fill_random_edges(&mut self, num_edges: usize) {
				match &mut self.index {
					IndexOneOf::A(index) => fill_random_edges(index.graph_mut(), num_edges),
					IndexOneOf::B(index) => fill_random_edges(index.graph_mut(), num_edges),
					IndexOneOf::None => panic!(),
				}
			}
		}
	};
}


#[pyclass]
pub struct PyHNSW {
	index: IndexOneOf<GLIndex<ArrayView2<'static,f32>>, GCLIndex<ArrayView2<'static,f32>>>,
	max_frontier_size: Option<usize>,
	flooding: bool,
}
#[pymethods]
impl PyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, higher_level_max_heap_size=None, flooding=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		post_prune_heuristic: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
		finetune_rnn: Option<bool>,
		finetune_sen: Option<bool>,
		max_frontier_size: Option<usize>,
		higher_level_max_heap_size: Option<usize>,
		flooding: Option<bool>,
	) -> Self {
		let hnsw_params = HNSWParams::new()
		.maybe_with_higher_max_degree(higher_max_degree)
		.maybe_with_lowest_max_degree(lowest_max_degree)
		.maybe_with_max_layers(max_layers)
		.maybe_with_n_parallel_burnin(n_parallel_burnin)
		.maybe_with_max_build_heap_size(max_build_heap_size)
		.with_max_build_frontier_size(max_build_frontier_size)
		.with_level_norm_param_override(level_norm_param_override)
		.maybe_with_insert_heuristic(insert_heuristic)
		.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
		.maybe_with_post_prune_heuristic(post_prune_heuristic)
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		.maybe_with_finetune_rnn(finetune_rnn)
		.maybe_with_finetune_sen(finetune_sen)
		;
		unsafe {
			if flooding.unwrap_or(false) {
				let index = FloodingHNSWBuilder::build(
					arrview2_py_to_rust(data.as_array()),
					SquaredEuclideanDistance::new(),
					hnsw_params,
					higher_level_max_heap_size.unwrap_or(1),
				);
				if max_frontier_size.is_some() {
					let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
					PyHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size, flooding: true }
				} else {
					PyHNSW { index: IndexOneOf::A(index), max_frontier_size: None, flooding: true }
				}
			} else {
				let index = HNSWParallelHeapBuilder::build(
					arrview2_py_to_rust(data.as_array()),
					SquaredEuclideanDistance::new(),
					hnsw_params,
					higher_level_max_heap_size.unwrap_or(1),
				);
				if max_frontier_size.is_some() {
					let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
					PyHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size, flooding: false }
				} else {
					PyHNSW { index: IndexOneOf::A(index), max_frontier_size: None, flooding: false }
				}
			}
		}
	}
	#[getter]
	fn get_flooding(&self) -> bool {
		self.flooding
	}
}
generic_graph_index_funs!(layered PyHNSW);
#[pyclass]
pub struct PyFatHNSW {
	index: IndexOneOf<FGLIndex<ArrayView2<'static,f32>>, FGCLIndex<ArrayView2<'static,f32>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyFatHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, higher_level_max_heap_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		post_prune_heuristic: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
		finetune_rnn: Option<bool>,
		finetune_sen: Option<bool>,
		max_frontier_size: Option<usize>,
		higher_level_max_heap_size: Option<usize>,
	) -> Self {
		let hnsw_params = HNSWParams::new()
		.maybe_with_higher_max_degree(higher_max_degree)
		.maybe_with_lowest_max_degree(lowest_max_degree)
		.maybe_with_max_layers(max_layers)
		.maybe_with_n_parallel_burnin(n_parallel_burnin)
		.maybe_with_max_build_heap_size(max_build_heap_size)
		.with_max_build_frontier_size(max_build_frontier_size)
		.with_level_norm_param_override(level_norm_param_override)
		.maybe_with_insert_heuristic(insert_heuristic)
		.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
		.maybe_with_post_prune_heuristic(post_prune_heuristic)
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		.maybe_with_finetune_rnn(finetune_rnn)
		.maybe_with_finetune_sen(finetune_sen)
		;
		unsafe {
			let index = HNSWParallelHeapBuilder::build_fat(
				arrview2_py_to_rust(data.as_array()),
				SquaredEuclideanDistance::new(),
				hnsw_params,
				higher_level_max_heap_size.unwrap_or(1),
			);
			if max_frontier_size.is_some() {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				PyFatHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size }
			} else {
				PyFatHNSW { index: IndexOneOf::A(index), max_frontier_size: None }
			}
		}
	}
}
generic_graph_index_funs!(layered PyFatHNSW);
#[pyclass]
pub struct OwningPyHNSW {
	index: IndexOneOf<GLIndex<Array2<f32>>, GCLIndex<Array2<f32>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl OwningPyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, higher_level_max_heap_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		post_prune_heuristic: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
		finetune_rnn: Option<bool>,
		finetune_sen: Option<bool>,
		max_frontier_size: Option<usize>,
		higher_level_max_heap_size: Option<usize>,
	) -> Self {
		let hnsw_params = HNSWParams::new()
		.maybe_with_higher_max_degree(higher_max_degree)
		.maybe_with_lowest_max_degree(lowest_max_degree)
		.maybe_with_max_layers(max_layers)
		.maybe_with_n_parallel_burnin(n_parallel_burnin)
		.maybe_with_max_build_heap_size(max_build_heap_size)
		.with_max_build_frontier_size(max_build_frontier_size)
		.with_level_norm_param_override(level_norm_param_override)
		.maybe_with_insert_heuristic(insert_heuristic)
		.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
		.maybe_with_post_prune_heuristic(post_prune_heuristic)
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		.maybe_with_finetune_rnn(finetune_rnn)
		.maybe_with_finetune_sen(finetune_sen)
		;
		unsafe {
			let index = HNSWParallelHeapBuilder::build(
				arrview2_py_to_rust(data.as_array()).into_owned(),
				SquaredEuclideanDistance::new(),
				hnsw_params,
				higher_level_max_heap_size.unwrap_or(1),
			);
			if max_frontier_size.is_some() {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				OwningPyHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size }
			} else {
				OwningPyHNSW { index: IndexOneOf::A(index), max_frontier_size: None }
			}
		}
	}
}
generic_graph_index_funs!(layered OwningPyHNSW);
#[pyclass]
pub struct OwningPyFatHNSW {
	index: IndexOneOf<FGLIndex<Array2<f32>>, FGCLIndex<Array2<f32>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl OwningPyFatHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		post_prune_heuristic: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
		finetune_rnn: Option<bool>,
		finetune_sen: Option<bool>,
		max_frontier_size: Option<usize>,
	) -> Self {
		let hnsw_params = HNSWParams::new()
		.maybe_with_higher_max_degree(higher_max_degree)
		.maybe_with_lowest_max_degree(lowest_max_degree)
		.maybe_with_max_layers(max_layers)
		.maybe_with_n_parallel_burnin(n_parallel_burnin)
		.maybe_with_max_build_heap_size(max_build_heap_size)
		.with_max_build_frontier_size(max_build_frontier_size)
		.with_level_norm_param_override(level_norm_param_override)
		.maybe_with_insert_heuristic(insert_heuristic)
		.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
		.maybe_with_post_prune_heuristic(post_prune_heuristic)
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		.maybe_with_finetune_rnn(finetune_rnn)
		.maybe_with_finetune_sen(finetune_sen)
		;
		unsafe {
			let index = HNSWParallelHeapBuilder::build_fat(
				arrview2_py_to_rust(data.as_array()).into_owned(),
				SquaredEuclideanDistance::new(),
				hnsw_params,
				1,
			);
			if max_frontier_size.is_some() {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				OwningPyFatHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size }
			} else {
				OwningPyFatHNSW { index: IndexOneOf::A(index), max_frontier_size: None }
			}
		}
	}
}
generic_graph_index_funs!(layered OwningPyFatHNSW);

#[pyclass]
pub struct PySENHNSW {
	index: IndexOneOf<GLIndex<ArrayView2<'static,f32>>, GCLIndex<ArrayView2<'static,f32>>>,
	max_frontier_size: Option<usize>,
	flooding: bool,
}
#[pymethods]
impl PySENHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, max_cos=None, higher_level_max_heap_size=None, flooding=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		post_prune_heuristic: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
		finetune_rnn: Option<bool>,
		finetune_sen: Option<bool>,
		max_frontier_size: Option<usize>,
		max_cos: Option<f64>,
		higher_level_max_heap_size: Option<usize>,
		flooding: Option<bool>,
	) -> Self {
		let hnsw_params = HNSWSENParams::new()
		.maybe_with_higher_max_degree(higher_max_degree)
		.maybe_with_lowest_max_degree(lowest_max_degree)
		.maybe_with_max_layers(max_layers)
		.maybe_with_n_parallel_burnin(n_parallel_burnin)
		.maybe_with_max_build_heap_size(max_build_heap_size)
		.with_max_build_frontier_size(max_build_frontier_size)
		.with_level_norm_param_override(level_norm_param_override)
		.maybe_with_insert_heuristic(insert_heuristic)
		.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
		.maybe_with_post_prune_heuristic(post_prune_heuristic)
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		.maybe_with_finetune_rnn(finetune_rnn)
		.maybe_with_finetune_sen(finetune_sen)
		.with_finetune_sen_params(SENParams::new()
			.maybe_with_max_cos(max_cos.map(|v| v as f32))
		)
		.maybe_with_max_cos(max_cos)
		;
		unsafe {
			if flooding.unwrap_or(false) {
				let index = FloodingHNSWSENBuilder::build(
					arrview2_py_to_rust(data.as_array()),
					SquaredEuclideanDistance::new(),
					hnsw_params,
					higher_level_max_heap_size.unwrap_or(1),
				);
				if max_frontier_size.is_some() {
					let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
					PySENHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size, flooding: false }
				} else {
					PySENHNSW { index: IndexOneOf::A(index), max_frontier_size: None, flooding: false }
				}
			} else {
				let index = HNSWParallelSENHeapBuilder::build(
					arrview2_py_to_rust(data.as_array()),
					SquaredEuclideanDistance::new(),
					hnsw_params,
					higher_level_max_heap_size.unwrap_or(1),
				);
				if max_frontier_size.is_some() {
					let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
					PySENHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size, flooding: true }
				} else {
					PySENHNSW { index: IndexOneOf::A(index), max_frontier_size: None, flooding: true }
				}
			}
		}
	}
	#[getter]
	fn get_flooding(&self) -> bool {
		self.flooding
	}
}
generic_graph_index_funs!(layered PySENHNSW);

#[pyclass]
pub struct PyRNNDescent {
	index: IndexOneOf<GSIndex<ArrayView2<'static,f32>>, GCSIndex<ArrayView2<'static,f32>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyRNNDescent {
	#[new]
	#[pyo3(signature = (data, initial_degree=None, reduce_degree=None, n_outer_loops=None, n_inner_loops=None, concurrent_batch_size=None, max_frontier_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		initial_degree: Option<usize>,
		reduce_degree: Option<usize>,
		n_outer_loops: Option<usize>,
		n_inner_loops: Option<usize>,
		concurrent_batch_size: Option<usize>,
		max_frontier_size: Option<usize>,
	) -> Self {
		let params = crate::rnn::RNNParams::new()
		.maybe_with_initial_degree(initial_degree)
		.maybe_with_reduce_degree(reduce_degree)
		.maybe_with_n_outer_loops(n_outer_loops)
		.maybe_with_n_inner_loops(n_inner_loops)
		.maybe_with_concurrent_batch_size(concurrent_batch_size)
		;
		unsafe {
			let index = crate::rnn::RNNDescentBuilder::build(
				arrview2_py_to_rust(data.as_array()),
				SquaredEuclideanDistance::new(),
				params,
			);
			if max_frontier_size.is_none() {
				PyRNNDescent { index: IndexOneOf::A(index), max_frontier_size: None }
			} else {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				PyRNNDescent { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size }
			}
		}
	}
}
generic_graph_index_funs!(single PyRNNDescent);
#[pyclass]
pub struct PySENDescent {
	index: IndexOneOf<GSIndex<ArrayView2<'static,f32>>, GCSIndex<ArrayView2<'static,f32>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PySENDescent {
	#[new]
	#[pyo3(signature = (data, initial_degree=None, reduce_degree=None, n_outer_loops=None, n_inner_loops=None, concurrent_batch_size=None, max_cos=None, dist_is_sq=None, prune_non_sen_edges=None, verify_sen_edges=None, max_frontier_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		initial_degree: Option<usize>,
		reduce_degree: Option<usize>,
		n_outer_loops: Option<usize>,
		n_inner_loops: Option<usize>,
		concurrent_batch_size: Option<usize>,
		max_cos: Option<f32>,
		dist_is_sq: Option<bool>,
		prune_non_sen_edges: Option<bool>,
		verify_sen_edges: Option<bool>,
		max_frontier_size: Option<usize>,
	) -> Self {
		let params = crate::rnn::SENParams::new()
		.maybe_with_initial_degree(initial_degree)
		.maybe_with_reduce_degree(reduce_degree)
		.maybe_with_n_outer_loops(n_outer_loops)
		.maybe_with_n_inner_loops(n_inner_loops)
		.maybe_with_concurrent_batch_size(concurrent_batch_size)
		.maybe_with_max_cos(max_cos)
		.maybe_with_dist_is_sq(dist_is_sq)
		.maybe_with_prune_non_sen_edges(prune_non_sen_edges)
		.maybe_with_verify_sen_edges(verify_sen_edges)
		;
		unsafe {
			let index = crate::rnn::SENDescentBuilder::build(
				arrview2_py_to_rust(data.as_array()),
				SquaredEuclideanDistance::new(),
				params,
			);
			if max_frontier_size.is_none() {
				PySENDescent { index: IndexOneOf::A(index), max_frontier_size: None }
			} else {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				PySENDescent { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size }
			}
		}
	}
}
generic_graph_index_funs!(single PySENDescent);


#[pyfunction]
#[pyo3(signature = (file, max_frontier_size=None))]
pub fn load_hnswlib(file: &str, max_frontier_size: Option<usize>) -> OwningPyHNSW {
	let index = crate::hnsw::load_hnswlib(file);
	if max_frontier_size.is_none() {
		OwningPyHNSW{index:IndexOneOf::A(index), max_frontier_size:None}
	} else {
		OwningPyHNSW{index:IndexOneOf::B(index.into_capped(max_frontier_size.unwrap())), max_frontier_size:max_frontier_size}
	}
}
#[pyfunction]
#[pyo3(signature = (file, max_frontier_size=None))]
pub fn load_hnswlib_fat(file: &str, max_frontier_size: Option<usize>) -> OwningPyFatHNSW {
	let index = crate::hnsw::load_hnswlib_fat(file);
	if max_frontier_size.is_none() {
		OwningPyFatHNSW{index:IndexOneOf::A(index), max_frontier_size:None}
	} else {
		OwningPyFatHNSW{index:IndexOneOf::B(index.into_capped(max_frontier_size.unwrap())), max_frontier_size:max_frontier_size}
	}
}

#[pyfunction]
#[pyo3(signature = (data, min_pts, expand=None, symmetric_expand=None, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None))]
pub fn hnsw_based_dendrogram<'py>(
	data: Bound<'py, PyArray2<f32>>,
	min_pts: usize,
	expand: Option<bool>,
	symmetric_expand: Option<bool>,
	higher_max_degree: Option<usize>,
	lowest_max_degree: Option<usize>,
	max_layers: Option<usize>,
	n_parallel_burnin: Option<usize>,
	max_build_heap_size: Option<usize>,
	max_build_frontier_size: Option<usize>,
	level_norm_param_override: Option<f32>,
	insert_heuristic: Option<bool>,
	insert_heuristic_extend: Option<bool>,
	post_prune_heuristic: Option<bool>,
	insert_minibatch_size: Option<usize>,
	n_rounds: Option<usize>,
) -> (Vec<(usize, usize, f32, usize)>, Vec<f32>) {
	let hnsw_params = HNSWParams::new()
	.maybe_with_higher_max_degree(higher_max_degree)
	.maybe_with_lowest_max_degree(lowest_max_degree)
	.maybe_with_max_layers(max_layers)
	.maybe_with_n_parallel_burnin(n_parallel_burnin)
	.maybe_with_max_build_heap_size(max_build_heap_size)
	.with_max_build_frontier_size(max_build_frontier_size)
	.with_level_norm_param_override(level_norm_param_override)
	.maybe_with_insert_heuristic(insert_heuristic)
	.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
	.maybe_with_post_prune_heuristic(post_prune_heuristic)
	.maybe_with_insert_minibatch_size(insert_minibatch_size)
	.maybe_with_n_rounds(n_rounds)
	;
	unsafe {
		let mut result = crate::cluster::hnsw_based_dendrogram::<f32,usize,_,_>(
			&arrview2_py_to_rust(data.as_array()),
			SquaredEuclideanDistance::new(),
			min_pts,
			expand.unwrap_or(true),
			symmetric_expand.unwrap_or(false),
			hnsw_params,
		);
		result.0.iter_mut().for_each(|(_,_,d,_)| *d = d.max(0.0).sqrt());
		result.1.iter_mut().for_each(|d| *d = d.max(0.0).sqrt());
		result
	}
}

#[pyfunction]
#[pyo3(signature = (data, min_pts, self_join_neighbors, query_max_heap_size, expand=None, symmetric_expand=None, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, query_local=false))]
pub fn hnsw_based_dendrogram_self_joined<'py>(
	data: Bound<'py, PyArray2<f32>>,
	min_pts: usize,
	self_join_neighbors: usize,
	query_max_heap_size: usize,
	expand: Option<bool>,
	symmetric_expand: Option<bool>,
	higher_max_degree: Option<usize>,
	lowest_max_degree: Option<usize>,
	max_layers: Option<usize>,
	n_parallel_burnin: Option<usize>,
	max_build_heap_size: Option<usize>,
	max_build_frontier_size: Option<usize>,
	level_norm_param_override: Option<f32>,
	insert_heuristic: Option<bool>,
	insert_heuristic_extend: Option<bool>,
	post_prune_heuristic: Option<bool>,
	insert_minibatch_size: Option<usize>,
	n_rounds: Option<usize>,
	query_local: Option<bool>,
) -> (Vec<(usize, usize, f32, usize)>, Vec<f32>) {
	let hnsw_params = HNSWParams::new()
	.maybe_with_higher_max_degree(higher_max_degree)
	.maybe_with_lowest_max_degree(lowest_max_degree)
	.maybe_with_max_layers(max_layers)
	.maybe_with_n_parallel_burnin(n_parallel_burnin)
	.maybe_with_max_build_heap_size(max_build_heap_size)
	.with_max_build_frontier_size(max_build_frontier_size)
	.with_level_norm_param_override(level_norm_param_override)
	.maybe_with_insert_heuristic(insert_heuristic)
	.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
	.maybe_with_post_prune_heuristic(post_prune_heuristic)
	.maybe_with_insert_minibatch_size(insert_minibatch_size)
	.maybe_with_n_rounds(n_rounds)
	;
	unsafe {
		let mut result = crate::cluster::hnsw_based_dendrogram_self_joined::<f32,usize,_,_>(
			&arrview2_py_to_rust(data.as_array()),
			SquaredEuclideanDistance::new(),
			min_pts,
			expand.unwrap_or(true),
			symmetric_expand.unwrap_or(false),
			hnsw_params,
			self_join_neighbors,
			query_max_heap_size,
			query_local.unwrap_or(false),
		);
		result.0.iter_mut().for_each(|(_,_,d,_)| *d = d.max(0.0).sqrt());
		result.1.iter_mut().for_each(|d| *d = d.max(0.0).sqrt());
		result
	}
}

#[pymodule(name="graphidxbaselines")]
fn hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<PyHNSW>()?;
	m.add_class::<PyFatHNSW>()?;
	m.add_class::<OwningPyHNSW>()?;
	m.add_class::<OwningPyFatHNSW>()?;
	m.add_class::<PySENHNSW>()?;
	m.add_class::<PyRNNDescent>()?;
	m.add_class::<PySENDescent>()?;
	m.add_function(wrap_pyfunction!(load_hnswlib, m)?)?;
	m.add_function(wrap_pyfunction!(load_hnswlib_fat, m)?)?;
	m.add_function(wrap_pyfunction!(hnsw_based_dendrogram, m)?)?;
	m.add_function(wrap_pyfunction!(hnsw_based_dendrogram_self_joined, m)?)?;
	Ok(())
}


