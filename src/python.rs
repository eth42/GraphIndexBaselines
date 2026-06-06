

// struct PyArrayDataSource

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use graphidx::{data::{MatrixDataSource, InterleavedSparseMatrix, TransmuteInto}, graph_ops::{extend_random_edges, fill_random_edges}, graphs::{DirLoLGraph, FatDirGraph, Graph, WDirLoLGraph, WeightedGraph}, indices::{GraphIndex, GreedyCappedLayeredGraphIndex, GreedyCappedSingleGraphIndex, GreedyLayeredGraphIndex, GreedySingleGraphIndex}, measures::Distance, types::{SyncFloat, UnsignedInteger}};
#[allow(unused)]
use num::traits::ConstZero;
use num::NumCast;
use pyo3::prelude::*;
use paste::paste;

use crate::{hnsw::{FloodingHNSWBuilder, FloodingHNSWSENBuilder, HNSWParallelHeapBuilder, HNSWParallelSENHeapBuilder, HNSWParams, HNSWSENParams, HNSWStyleBuilder}, rnn::{RNNStyleBuilder, SENParams}};

/* Size of references (IDs) in wrapped indices */
macro_rules! make_reference_types {
	($utype: ty, $itype: ty, $bits: literal) => {
		mod reference_types {
			pub type PyUint = $utype;
			pub type PyInt = $itype;
			pub const REF_BITS: usize = $bits;
		}
	};
	($($feature: literal: $bits: literal),*$(,)*) => {
		paste! {
			$(
				#[cfg(feature=$feature)]
				make_reference_types!([<u $bits>], [<i $bits>], $bits);
			)*
			#[cfg(not(any($(feature=$feature,)*)))]
			#[cfg(target_pointer_width = "32")]
			make_reference_types!(usize, isize, 32);
			#[cfg(not(any($(feature=$feature,)*)))]
			#[cfg(target_pointer_width = "64")]
			make_reference_types!(usize, isize, 64);
		}
		type PyUint = reference_types::PyUint;
		type PyInt = reference_types::PyInt;
		const REF_BITS: usize = reference_types::REF_BITS;
		#[pyfunction]
		fn ref_bits() -> usize { REF_BITS }
	};
}
make_reference_types!(
	"pyref16": 16,
	"pyref32": 32,
	"pyref64": 64,
	"pyref128": 128,
);
/* Numerical precision */
#[allow(non_camel_case_types)]
#[allow(unused)]
type f16 = half::f16;
macro_rules! make_precision_types {
	($ftype: ty, $itype: ty, $utype: ty, $bits: literal) => {
		mod precision_types {
			pub type PyFloat = $ftype;
			/* int and uint type with equivalent number of bytes to PyFloat for interleaved storage */
			pub type PyFloatInt = $itype;
			pub type PyFloatUint = $utype;
			pub const PREC_BITS: usize = $bits;
		}
	};
	($($feature: literal: ($ftype: path, $itype: ty, $utype: ty, $bits: literal)),*$(,)*) => {
		paste! {
			$(
				#[cfg(feature=$feature)]
				make_precision_types!($ftype, $itype, $utype, $bits);
			)*
			#[cfg(not(any($(feature=$feature,)*)))]
			#[cfg(target_pointer_width = "32")]
			make_precision_types!(f32, i32, u32, 32);
			#[cfg(not(any($(feature=$feature,)*)))]
			#[cfg(target_pointer_width = "64")]
			make_precision_types!(f64, i64, u64, 64);
		}
		type PyFloat = precision_types::PyFloat;
		type PyFloatInt = precision_types::PyFloatInt;
		type PyFloatUint = precision_types::PyFloatUint;
		const PREC_BITS: usize = precision_types::PREC_BITS;
		#[pyfunction]
		fn prec_bits() -> usize { PREC_BITS }
	};
}
make_precision_types!(
	"pyprec16": (half::f16, i16, u16, 16),
	"pyprec32": (f32, i32, u32, 32),
	"pyprec64": (f64, i64, u64, 64),
	// "pyprec128": (f128, i128, u128, 128),
);

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


pub trait DistanceWrapper<F: SyncFloat+numpy::Element> {
	type Dist: Distance<F>;
	fn get_dist(&self) -> &Self::Dist;
	fn dist<'py>(&self, u: Bound<'py, PyArray1<F>>, v: Bound<'py, PyArray1<F>>) -> F {
		unsafe { self.get_dist().dist_slice(u.as_slice().unwrap(), v.as_slice().unwrap()) }
	}
	fn to_enum(self) -> DistanceEnum;
}
macro_rules! make_distance_wrapper {
	(normal $name: ident $(($($arg: ident),*$(,)?))?) => {
		paste! {
			#[derive(Clone)]
			#[pyclass]
			pub struct $name {
				dist: graphidx::measures::$name<PyFloat>,
			}
			#[pymethods]
			impl $name {
				#[new]
				#[pyo3(signature = ($($($arg,)*)?))]
				fn new<'py>($($($arg: f64,)*)?) -> Self {
					$name { dist: graphidx::measures::$name::new($($(<PyFloat as NumCast>::from($arg).unwrap(),)*)?)}
				}
				fn dist<'py>(&self, u: Bound<'py, PyArray1<PyFloat>>, v: Bound<'py, PyArray1<PyFloat>>) -> f64 {
					<f64 as NumCast>::from(DistanceWrapper::<PyFloat>::dist(self, u, v)).unwrap()
				}
				fn to_enum(&self) -> DistanceEnum {
					DistanceWrapper::<PyFloat>::to_enum(self.clone())
				}
			}
			impl DistanceWrapper<PyFloat> for $name {
				type Dist = graphidx::measures::$name<PyFloat>;
				fn get_dist(&self) -> &graphidx::measures::$name<PyFloat> {
					&self.dist
				}
				fn to_enum(self) -> DistanceEnum {
					DistanceEnum::$name(self)
				}
			}
			impl Into<DistanceEnum> for $name {
				fn into(self) -> DistanceEnum {
					self.to_enum()
				}
			}
		}
	};
	(sparse $name: ident $(($($arg: ident),*$(,)?))?) => {
		paste! {
			#[derive(Clone)]
			#[pyclass]
			pub struct $name {
				dist: graphidx::measures::$name<PyFloat,PyFloatUint>,
			}
			#[pymethods]
			impl $name {
				#[new]
				#[pyo3(signature = ($($($arg,)*)?))]
				fn new<'py>($($($arg:f64,)*)?) -> Self {
					$name { dist: graphidx::measures::$name::new($($(<PyFloat as NumCast>::from($arg).unwrap(),)*)?) }
				}
				fn dist<'py>(&self, u: Bound<'py, PyArray1<PyFloat>>, v: Bound<'py, PyArray1<PyFloat>>) -> f64 {
					<f64 as NumCast>::from(DistanceWrapper::<PyFloat>::dist(self, u, v)).unwrap()
				}
				fn to_enum(&self) -> DistanceEnum {
					DistanceWrapper::<PyFloat>::to_enum(self.clone())
				}
			}
			impl DistanceWrapper<PyFloat> for $name {
				type Dist = graphidx::measures::$name<PyFloat,PyFloatUint>;
				fn get_dist(&self) -> &graphidx::measures::$name<PyFloat,PyFloatUint> {
					&self.dist
				}
				fn to_enum(self) -> DistanceEnum {
					DistanceEnum::$name(self)
				}
			}
			impl Into<DistanceEnum> for $name {
				fn into(self) -> DistanceEnum {
					self.to_enum()
				}
			}
		}
	};
	(
		$(
			$label: ident [
				$(
					$name: ident$(($($arg: ident),*$(,)?))?
				),*$(,)?
			]
		),*$(,)?
	) => {	
		#[derive(Clone)]
		#[pyclass]
		pub enum DistanceEnum {
			$($($name($name)),*),*
		}
		impl Distance<PyFloat> for DistanceEnum {
			#[inline(always)]
			fn dist_slice(&self, obj1: &[PyFloat], obj2: &[PyFloat]) -> PyFloat {
				match self {
					$($(
						Self::$name(d) => d.get_dist().dist_slice(obj1,obj2)
					),*),*
				}
			}
		}
		$($(make_distance_wrapper!($label $name$(($($arg,)*))?);)*)?
		macro_rules! add_distance_wrappers_to_module {
			($module: ident) => {
				$($(
					$module.add_class::<$name>()?;
				)*)*
			};
		}
	};
}
make_distance_wrapper!(
	normal [
		SquaredEuclideanDistance, EuclideanDistance, NegDotProduct, CosineDistance, HammingDistance, DotProdSurrogateAdd, DotProdSurrogateSub, DotProdSurrogateMix(factor),
	],
	sparse [
		SparseSquaredEuclideanDistance, SparseEuclideanDistance, SparseNormedSquaredEuclideanDistance, SparseNegDotProduct, SparseDotProdSurrogateAdd, SparseDotProdSurrogateSub, SparseDotProdSurrogateMix(factor),
	],
);

type GSIndex<M> = GreedySingleGraphIndex<PyUint, PyFloat, DistanceEnum, M, DirLoLGraph<PyUint>>;
type GCSIndex<M> = GreedyCappedSingleGraphIndex<PyUint, PyFloat, DistanceEnum, M, DirLoLGraph<PyUint>>;
type GSWIndex<M> = GreedySingleGraphIndex<PyUint, PyFloat, DistanceEnum, M, WDirLoLGraph<PyUint, PyFloat>>;
type GCSWIndex<M> = GreedyCappedSingleGraphIndex<PyUint, PyFloat, DistanceEnum, M, WDirLoLGraph<PyUint, PyFloat>>;
// type FGSIndex<M> = GreedySingleGraphIndex<PyUint, PyFloat, DistanceEnum, M, FatDirGraph<PyUint>>;
// type FGCSIndex<M> = GreedyCappedSingleGraphIndex<PyUint, PyFloat, DistanceEnum, M, FatDirGraph<PyUint>>;
type GLIndex<M> = GreedyLayeredGraphIndex<PyUint, PyFloat, DistanceEnum, M, DirLoLGraph<PyUint>>;
type GCLIndex<M> = GreedyCappedLayeredGraphIndex<PyUint, PyFloat, DistanceEnum, M, DirLoLGraph<PyUint>>;
type FGLIndex<M> = GreedyLayeredGraphIndex<PyUint, PyFloat, DistanceEnum, M, FatDirGraph<PyUint>>;
type FGCLIndex<M> = GreedyCappedLayeredGraphIndex<PyUint, PyFloat, DistanceEnum, M, FatDirGraph<PyUint>>;


pub enum IndexOneOf<A: GraphIndex<PyUint,PyFloat,DistanceEnum>, B: GraphIndex<PyUint,PyFloat,DistanceEnum>> {
	A(A),
	B(B),
	None,
}
#[allow(dead_code)]
impl<A: GraphIndex<PyUint,PyFloat,DistanceEnum>, B: GraphIndex<PyUint,PyFloat,DistanceEnum>> IndexOneOf<A,B> {
	fn greedy_search(&self, queries: &ArrayView1<PyFloat>, k: usize, max_heap_size: usize) -> (Array1<PyUint>, Array1<PyFloat>) {
		match self {
				IndexOneOf::A(a) => a.greedy_search(queries, k, max_heap_size, &mut a._new_search_cache(max_heap_size)),
				IndexOneOf::B(b) => b.greedy_search(queries, k, max_heap_size, &mut b._new_search_cache(max_heap_size)),
				IndexOneOf::None => panic!(),
		}
	}
	fn greedy_search_batch<M: MatrixDataSource<PyFloat>+Sync>(&self, queries: &M, k: usize, max_heap_size: usize) -> (Array2<PyUint>, Array2<PyFloat>) {
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
		generic_graph_index_funs!(_basic $type);
		#[pymethods]
		impl $type {
			#[pyo3(signature = (distance, data))]
			fn with_distance_and_data<'py>(&mut self, distance: DistanceEnum, data: Bound<'py, PyArray2<PyFloat>>) {
				let data = unsafe { arrview2_py_to_rust(data.as_array()) };
				let mut index_buffer = IndexOneOf::None;
				std::mem::swap(&mut index_buffer, &mut self.index);
				self.index = match index_buffer {
					IndexOneOf::A(index) => IndexOneOf::A(index.with_distance_and_data(distance, data)),
					IndexOneOf::B(index) => IndexOneOf::B(index.with_distance_and_data(distance, data)),
					IndexOneOf::None => panic!(),
				};
			}
		}
	};
	(layered $type: ident) => {
		generic_graph_index_funs!($type);
		generic_graph_index_funs!(_layered $type);
	};
	(single $type: ident) => {
		generic_graph_index_funs!($type);
		generic_graph_index_funs!(_single $type);
	};
	(owning $type: ident) => {
		generic_graph_index_funs!(_basic $type);
		#[pymethods]
		impl $type {
			#[pyo3(signature = (distance, data))]
			fn with_distance_and_data<'py>(&mut self, distance: DistanceEnum, data: Bound<'py, PyArray2<PyFloat>>) {
				let data = unsafe { arrview2_py_to_rust(data.as_array()).into_owned() };
				let mut index_buffer = IndexOneOf::None;
				std::mem::swap(&mut index_buffer, &mut self.index);
				self.index = match index_buffer {
					IndexOneOf::A(index) => IndexOneOf::A(index.with_distance_and_data(distance, data)),
					IndexOneOf::B(index) => IndexOneOf::B(index.with_distance_and_data(distance, data)),
					IndexOneOf::None => panic!(),
				};
			}
		}
	};
	(owning layered $type: ident) => {
		generic_graph_index_funs!(owning $type);
		generic_graph_index_funs!(_layered $type);
	};
	(owning single $type: ident) => {
		generic_graph_index_funs!(owning $type);
		generic_graph_index_funs!(_single $type);
	};
	(_basic $type: ident) => {
		#[pymethods]
		impl $type {
			#[pyo3(signature = (query, k, max_heap_size=None))]
			fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<PyFloat>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<PyUint>>, Bound<'py,PyArray1<PyFloat>>) {
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
			fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<PyFloat>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<PyUint>>, Bound<'py,PyArray2<PyFloat>>) {
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
			#[pyo3(signature = (distance))]
			fn with_distance(&mut self, distance: DistanceEnum) {
				let mut index_buffer = IndexOneOf::None;
				std::mem::swap(&mut index_buffer, &mut self.index);
				self.index = match index_buffer {
					IndexOneOf::A(index) => IndexOneOf::A(index.with_distance(distance)),
					IndexOneOf::B(index) => IndexOneOf::B(index.with_distance(distance)),
					IndexOneOf::None => panic!(),
				};
			}
			#[pyo3(signature = (k, max_heap_size, slice=None))]
			fn self_join_query(&self, k: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> PySelfJoinGraph {
				match &self.index {
					IndexOneOf::A(index) => {
						let graph = index.self_join_query_slice(k, max_heap_size, slice);
						PySelfJoinGraph {
							index: IndexOneOf::A(GSWIndex {
								_phantom: std::marker::PhantomData,
								data: unsafe {std::mem::transmute(index.data.view())},
								graph: graph,
								distance: index.distance.clone(),
								entry_points: None,
							}),
							max_frontier_size: None,
						}
					},
					IndexOneOf::B(index) => {
						let graph = index.self_join_query_slice(k, max_heap_size, slice);
						PySelfJoinGraph {
							index: IndexOneOf::A(GSWIndex {
								_phantom: std::marker::PhantomData,
								data: unsafe {std::mem::transmute(index.data.view())},
								graph: graph,
								distance: index.distance.clone(),
								entry_points: None,
							}),
							max_frontier_size: None,
						}
					},
					IndexOneOf::None => panic!(),
				}
			}
			#[pyo3(signature = (k, max_heap_size, slice=None))]
			fn self_join_query_local(&self, k: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> PySelfJoinGraph {
				match &self.index {
					IndexOneOf::A(index) => {
						let graph = index.self_join_query_local_slice(k, max_heap_size, slice);
						PySelfJoinGraph {
							index: IndexOneOf::A(GSWIndex {
								_phantom: std::marker::PhantomData,
								data: unsafe {std::mem::transmute(index.data.view())},
								graph: graph,
								distance: index.distance.clone(),
								entry_points: None,
							}),
							max_frontier_size: None,
						}
					},
					IndexOneOf::B(index) => {
						let graph = index.self_join_query_local_slice(k, max_heap_size, slice);
						PySelfJoinGraph {
							index: IndexOneOf::A(GSWIndex {
								_phantom: std::marker::PhantomData,
								data: unsafe {std::mem::transmute(index.data.view())},
								graph: graph,
								distance: index.distance.clone(),
								entry_points: None,
							}),
							max_frontier_size: None,
						}
					},
					IndexOneOf::None => panic!(),
				}
			}
			#[pyo3(signature = (k, max_heap_size, slice=None))]
			fn self_join_query_arr<'py>(&self, py: Python<'py>, k: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> (Bound<'py,PyArray2<PyUint>>, Bound<'py,PyArray2<PyFloat>>) {
				let (ids, dists) = match &self.index {
					IndexOneOf::A(index) => index.self_join_query_arr_slice(k, max_heap_size, slice),
					IndexOneOf::B(index) => index.self_join_query_arr_slice(k, max_heap_size, slice),
					IndexOneOf::None => panic!(),
				};
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			}
			#[pyo3(signature = (k, max_heap_size, slice=None))]
			fn self_join_query_local_arr<'py>(&self, py: Python<'py>, k: usize, max_heap_size: usize, slice: Option<(usize,usize)>) -> (Bound<'py,PyArray2<PyUint>>, Bound<'py,PyArray2<PyFloat>>) {
				let (ids, dists) = match &self.index {
					IndexOneOf::A(index) => index.self_join_query_local_arr_slice(k, max_heap_size, slice),
					IndexOneOf::B(index) => index.self_join_query_local_arr_slice(k, max_heap_size, slice),
					IndexOneOf::None => panic!(),
				};
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
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
	(_layered $type: ident) => {
		#[pymethods]
		impl $type {
			fn get_graph_stats(&self) -> Vec<GraphStats> {
				match &self.index {
					IndexOneOf::A(index) => index.graphs().iter().map(|g| GraphStats::from_graph(g)).collect(),
					IndexOneOf::B(index) => index.graphs().iter().map(|g| GraphStats::from_graph(g)).collect(),
					IndexOneOf::None => panic!(),
				}
			}
			fn get_neighbors(&self, layer: usize, node: PyUint) -> Vec<PyUint> {
				match &self.index {
					IndexOneOf::A(index) => index.graphs()[layer].neighbors(node),
					IndexOneOf::B(index) => index.graphs()[layer].neighbors(node),
					IndexOneOf::None => panic!(),
				}
			}
			fn get_next_layer_id(&self, layer: usize, node: PyUint) -> PyUint {
				if layer == 0 { return node; }
				match &self.index {
					IndexOneOf::A(index) => index.get_local_layer_ids(layer).unwrap()[node as usize],
					IndexOneOf::B(index) => index.get_local_layer_ids(layer).unwrap()[node as usize],
					IndexOneOf::None => panic!(),
				}
			}
			fn get_global_id(&self, layer: usize, node: PyUint) -> PyUint {
				if layer == 0 { return node; }
				match &self.index {
					IndexOneOf::A(index) => index.get_global_layer_ids(layer).unwrap()[node as usize],
					IndexOneOf::B(index) => index.get_global_layer_ids(layer).unwrap()[node as usize],
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
	(_single $type: ident) => {
		#[pymethods]
		impl $type {
			fn get_graph_stats(&self) -> GraphStats {
				match &self.index {
					IndexOneOf::A(index) => GraphStats::from_graph(index.graph()),
					IndexOneOf::B(index) => GraphStats::from_graph(index.graph()),
					IndexOneOf::None => panic!(),
				}
			}
			fn get_neighbors(&self, node: PyUint) -> Vec<PyUint> {
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
	(sparse $type: ident) => {
		#[pymethods]
		impl $type {
			#[pyo3(signature = (query_data, query_indices, k, max_heap_size=None))]
			fn knn_query<'py>(&self, py: Python<'py>, query_data: Bound<'py, PyArray1<PyFloat>>, query_indices: Bound<'py, PyArray1<PyFloatInt>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<PyUint>>, Bound<'py,PyArray1<PyFloat>>) {
				/* TODO: This should work on the (data,indices,indptr) tuple instead */
				unsafe {
					let data = arrview1_py_to_rust(query_data.as_array());
					let indices = arrview1_py_to_rust(query_indices.as_array());
					let interleaved = Array1::from_iter(
						indices.into_iter()
						.zip(data.into_iter())
						.map(|(&i,&v)| {
							std::iter::once(i.transmute()).chain(std::iter::once(v))
						}).flatten()
					);
					let (ids, dists) = self.index.greedy_search(
						&interleaved.view(),
						k,
						max_heap_size.unwrap_or(2*k),
					);
					(
						PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
						PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
					)
				}
			}
			#[pyo3(signature = (query_data, query_indices, query_indptr, k, max_heap_size=None))]
			fn knn_query_batch<'py>(&self, py: Python<'py>, query_data: Bound<'py, PyArray1<PyFloat>>, query_indices: Bound<'py, PyArray1<PyFloatInt>>, query_indptr: Bound<'py, PyArray1<PyInt>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<PyUint>>, Bound<'py,PyArray2<PyFloat>>) {
				unsafe {
					let data = arrview1_py_to_rust(query_data.as_array());
					let indices = arrview1_py_to_rust(query_indices.as_array());
					let indptr = arrview1_py_to_rust(query_indptr.as_array());
					let queries = InterleavedSparseMatrix::from_csr(data, indices, indptr, None);
					let (ids, dists) = self.index.greedy_search_batch(
						&queries,
						k,
						max_heap_size.unwrap_or(2*k),
					);
					(
						PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
						PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
					)
				}
			}
			#[pyo3(signature = (distance))]
			fn with_distance(&mut self, distance: DistanceEnum) {
				let mut index_buffer = IndexOneOf::None;
				std::mem::swap(&mut index_buffer, &mut self.index);
				self.index = match index_buffer {
					IndexOneOf::A(index) => IndexOneOf::A(index.with_distance(distance)),
					IndexOneOf::B(index) => IndexOneOf::B(index.with_distance(distance)),
					IndexOneOf::None => panic!(),
				};
			}
			#[pyo3(signature = (distance, data, indices, indptr))]
			fn with_distance_and_data<'py>(&mut self, distance: DistanceEnum, data: Bound<'py, PyArray1<PyFloat>>, indices: Bound<'py, PyArray1<PyFloatInt>>, indptr: Bound<'py, PyArray1<PyInt>>) {
				let mut index_buffer = IndexOneOf::None;
				let interleaved_data = unsafe {
					let data = arrview1_py_to_rust(data.as_array());
					let indices = arrview1_py_to_rust(indices.as_array());
					let indptr = arrview1_py_to_rust(indptr.as_array());
					InterleavedSparseMatrix::from_csr(data, indices, indptr, None)
				};
				std::mem::swap(&mut index_buffer, &mut self.index);
				self.index = match index_buffer {
					IndexOneOf::A(index) => IndexOneOf::A(index.with_distance_and_data(distance, interleaved_data)),
					IndexOneOf::B(index) => IndexOneOf::B(index.with_distance_and_data(distance, interleaved_data)),
					IndexOneOf::None => panic!(),
				};
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
}


#[pyclass]
pub struct PySelfJoinGraph {
	index: IndexOneOf<GSWIndex<ArrayView2<'static,PyFloat>>, GCSWIndex<ArrayView2<'static,PyFloat>>>,
	max_frontier_size: Option<usize>,
}
generic_graph_index_funs!(single PySelfJoinGraph);
#[pymethods]
impl PySelfJoinGraph {
	#[cfg(not(feature="pyprec16"))]
	fn get_neighbors_with_weight(&self, node: PyUint) -> Vec<(PyFloat,PyUint)> {
		match &self.index {
			IndexOneOf::A(index) => index.graph().neighbors_with_zipped_weights(node),
			IndexOneOf::B(index) => index.graph().neighbors_with_zipped_weights(node),
			IndexOneOf::None => panic!(),
		}
	}
	#[cfg(feature="pyprec16")]
	fn get_neighbors_with_weight(&self, node: PyUint) -> Vec<(f32,PyUint)> {
		match &self.index {
			IndexOneOf::A(index) => index.graph().neighbors_with_zipped_weights(node).into_iter().map(|(v,r)| (v.to_f32(),r)).collect(),
			IndexOneOf::B(index) => index.graph().neighbors_with_zipped_weights(node).into_iter().map(|(v,r)| (v.to_f32(),r)).collect(),
			IndexOneOf::None => panic!(),
		}
	}
}


#[pyclass]
pub struct PyHNSW {
	index: IndexOneOf<GLIndex<ArrayView2<'static,PyFloat>>, GCLIndex<ArrayView2<'static,PyFloat>>>,
	max_frontier_size: Option<usize>,
	flooding: bool,
}
#[pymethods]
impl PyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, higher_level_max_heap_size=None, flooding=None, distance=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<PyFloat>>,
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
		distance: Option<DistanceEnum>,
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
		let distance = distance.unwrap_or(DistanceEnum::SquaredEuclideanDistance(SquaredEuclideanDistance::new()));
		unsafe {
			if flooding.unwrap_or(false) {
				let index = FloodingHNSWBuilder::build(
					arrview2_py_to_rust(data.as_array()),
					distance,
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
					distance,
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
	index: IndexOneOf<FGLIndex<ArrayView2<'static,PyFloat>>, FGCLIndex<ArrayView2<'static,PyFloat>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyFatHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, higher_level_max_heap_size=None, distance=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<PyFloat>>,
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
		distance: Option<DistanceEnum>,
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
		let distance = distance.unwrap_or(DistanceEnum::SquaredEuclideanDistance(SquaredEuclideanDistance::new()));
		unsafe {
			let index = HNSWParallelHeapBuilder::build_fat(
				arrview2_py_to_rust(data.as_array()),
				distance,
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
	index: IndexOneOf<GLIndex<Array2<PyFloat>>, GCLIndex<Array2<PyFloat>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl OwningPyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, higher_level_max_heap_size=None, distance=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<PyFloat>>,
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
		distance: Option<DistanceEnum>,
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
		let distance = distance.unwrap_or(DistanceEnum::SquaredEuclideanDistance(SquaredEuclideanDistance::new()));
		unsafe {
			let index = HNSWParallelHeapBuilder::build(
				arrview2_py_to_rust(data.as_array()).into_owned(),
				distance,
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
generic_graph_index_funs!(owning layered OwningPyHNSW);
#[pyclass]
pub struct OwningPyFatHNSW {
	index: IndexOneOf<FGLIndex<Array2<PyFloat>>, FGCLIndex<Array2<PyFloat>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl OwningPyFatHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, distance=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<PyFloat>>,
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
		distance: Option<DistanceEnum>,
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
		let distance = distance.unwrap_or(DistanceEnum::SquaredEuclideanDistance(SquaredEuclideanDistance::new()));
		unsafe {
			let index = HNSWParallelHeapBuilder::build_fat(
				arrview2_py_to_rust(data.as_array()).into_owned(),
				distance,
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
generic_graph_index_funs!(owning layered OwningPyFatHNSW);

#[pyclass]
pub struct PySENHNSW {
	index: IndexOneOf<GLIndex<ArrayView2<'static,PyFloat>>, GCLIndex<ArrayView2<'static,PyFloat>>>,
	max_frontier_size: Option<usize>,
	flooding: bool,
}
#[pymethods]
impl PySENHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, max_cos=None, higher_level_max_heap_size=None, flooding=None, distance=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<PyFloat>>,
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
		distance: Option<DistanceEnum>,
	) -> Self {
		#[cfg(feature="pyprec16")]
		let sen_params = SENParams::new()
		.maybe_with_max_cos(max_cos.map(|v| PyFloat::from_f64(v)));
		#[cfg(not(feature="pyprec16"))]
		let sen_params = SENParams::new()
		.maybe_with_max_cos(max_cos.map(|v| v as PyFloat));
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
		.with_finetune_sen_params(sen_params)
		.maybe_with_max_cos(max_cos)
		;
		let distance = distance.unwrap_or(DistanceEnum::SquaredEuclideanDistance(SquaredEuclideanDistance::new()));
		unsafe {
			if flooding.unwrap_or(false) {
				let index = FloodingHNSWSENBuilder::build(
					arrview2_py_to_rust(data.as_array()),
					distance,
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
					distance,
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
	index: IndexOneOf<GSIndex<ArrayView2<'static,PyFloat>>, GCSIndex<ArrayView2<'static,PyFloat>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyRNNDescent {
	#[new]
	#[pyo3(signature = (data, initial_degree=None, reduce_degree=None, n_outer_loops=None, n_inner_loops=None, concurrent_batch_size=None, max_frontier_size=None, distance=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<PyFloat>>,
		initial_degree: Option<usize>,
		reduce_degree: Option<usize>,
		n_outer_loops: Option<usize>,
		n_inner_loops: Option<usize>,
		concurrent_batch_size: Option<usize>,
		max_frontier_size: Option<usize>,
		distance: Option<DistanceEnum>,
	) -> Self {
		let params = crate::rnn::RNNParams::new()
		.maybe_with_initial_degree(initial_degree)
		.maybe_with_reduce_degree(reduce_degree)
		.maybe_with_n_outer_loops(n_outer_loops)
		.maybe_with_n_inner_loops(n_inner_loops)
		.maybe_with_concurrent_batch_size(concurrent_batch_size)
		;
		let distance = distance.unwrap_or(DistanceEnum::SquaredEuclideanDistance(SquaredEuclideanDistance::new()));
		unsafe {
			let index = crate::rnn::RNNDescentBuilder::build(
				arrview2_py_to_rust(data.as_array()),
				distance,
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
	index: IndexOneOf<GSIndex<ArrayView2<'static,PyFloat>>, GCSIndex<ArrayView2<'static,PyFloat>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PySENDescent {
	#[new]
	#[pyo3(signature = (data, initial_degree=None, reduce_degree=None, n_outer_loops=None, n_inner_loops=None, concurrent_batch_size=None, max_cos=None, dist_is_sq=None, prune_non_sen_edges=None, verify_sen_edges=None, max_frontier_size=None, distance=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<PyFloat>>,
		initial_degree: Option<usize>,
		reduce_degree: Option<usize>,
		n_outer_loops: Option<usize>,
		n_inner_loops: Option<usize>,
		concurrent_batch_size: Option<usize>,
		max_cos: Option<f64>,
		dist_is_sq: Option<bool>,
		prune_non_sen_edges: Option<bool>,
		verify_sen_edges: Option<bool>,
		max_frontier_size: Option<usize>,
		distance: Option<DistanceEnum>,
	) -> Self {
		#[cfg(feature="pyprec16")]
		let max_cos = max_cos.map(|v| PyFloat::from_f64(v));
		#[cfg(not(feature="pyprec16"))]
		let max_cos = max_cos.map(|v| v as PyFloat);
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
		let distance = distance.unwrap_or(DistanceEnum::SquaredEuclideanDistance(SquaredEuclideanDistance::new()));
		unsafe {
			let index = crate::rnn::SENDescentBuilder::build(
				arrview2_py_to_rust(data.as_array()),
				distance,
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



#[pyclass]
pub struct SparsePyHNSW {
	index: IndexOneOf<GLIndex<InterleavedSparseMatrix<PyFloat>>, GCLIndex<InterleavedSparseMatrix<PyFloat>>>,
	max_frontier_size: Option<usize>,
	flooding: bool,
}
#[pymethods]
impl SparsePyHNSW {
	#[new]
	#[pyo3(signature = (data, indices, indptr, n_cols=None, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, finetune_rnn=None, finetune_sen=None, max_frontier_size=None, higher_level_max_heap_size=None, flooding=None, distance=None))]
	fn from_csr<'py>(
		data: Bound<'py, PyArray1<PyFloat>>,
		indices: Bound<'py, PyArray1<PyFloatInt>>,
		indptr: Bound<'py, PyArray1<PyInt>>,
		n_cols: Option<usize>,
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
		distance: Option<DistanceEnum>,
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
		let distance = distance.unwrap_or(DistanceEnum::SparseSquaredEuclideanDistance(SparseSquaredEuclideanDistance::new()));
		unsafe {
			let data = arrview1_py_to_rust(data.as_array());
			let indices = arrview1_py_to_rust(indices.as_array());
			let indptr = arrview1_py_to_rust(indptr.as_array());
			let mat = InterleavedSparseMatrix::from_csr(data, indices, indptr, n_cols);
			if flooding.unwrap_or(false) {
				let index = FloodingHNSWBuilder::build(
					mat,
					distance,
					hnsw_params,
					higher_level_max_heap_size.unwrap_or(1),
				);
				if max_frontier_size.is_some() {
					let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
					SparsePyHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size, flooding: true }
				} else {
					SparsePyHNSW { index: IndexOneOf::A(index), max_frontier_size: None, flooding: true }
				}
			} else {
				let index = HNSWParallelHeapBuilder::build(
					mat,
					distance,
					hnsw_params,
					higher_level_max_heap_size.unwrap_or(1),
				);
				if max_frontier_size.is_some() {
					let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
					SparsePyHNSW { index: IndexOneOf::B(capped_index), max_frontier_size: max_frontier_size, flooding: false }
				} else {
					SparsePyHNSW { index: IndexOneOf::A(index), max_frontier_size: None, flooding: false }
				}
			}
		}
	}
	#[getter]
	fn get_flooding(&self) -> bool {
		self.flooding
	}
}
generic_graph_index_funs!(sparse SparsePyHNSW);
generic_graph_index_funs!(_layered SparsePyHNSW);



#[pyfunction]
#[pyo3(signature = (file, max_frontier_size=None))]
pub fn load_hnswlib(file: &str, max_frontier_size: Option<usize>) -> OwningPyHNSW {
	let index = crate::hnsw::load_hnswlib(file);
	if max_frontier_size.is_none() {
		OwningPyHNSW{index:IndexOneOf::A(index.with_distance(SquaredEuclideanDistance::new().to_enum())), max_frontier_size:None}
	} else {
		OwningPyHNSW{index:IndexOneOf::B(index.into_capped(max_frontier_size.unwrap()).with_distance(SquaredEuclideanDistance::new().to_enum())), max_frontier_size:max_frontier_size}
	}
}
#[pyfunction]
#[pyo3(signature = (file, max_frontier_size=None))]
pub fn load_hnswlib_fat(file: &str, max_frontier_size: Option<usize>) -> OwningPyFatHNSW {
	let index = crate::hnsw::load_hnswlib_fat(file);
	if max_frontier_size.is_none() {
		OwningPyFatHNSW{index:IndexOneOf::A(index.with_distance(SquaredEuclideanDistance::new().to_enum())), max_frontier_size:None}
	} else {
		OwningPyFatHNSW{index:IndexOneOf::B(index.into_capped(max_frontier_size.unwrap()).with_distance(SquaredEuclideanDistance::new().to_enum())), max_frontier_size:max_frontier_size}
	}
}

#[cfg(feature="pyprec16")]
type DendrogramResult = (Vec<(usize, usize, f32, usize)>, Vec<f32>);
#[cfg(not(feature="pyprec16"))]
type DendrogramResult = (Vec<(usize, usize, PyFloat, usize)>, Vec<PyFloat>);
#[pyfunction]
#[pyo3(signature = (data, min_pts, expand=None, symmetric_expand=None, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None))]
pub fn hnsw_based_dendrogram<'py>(
	data: Bound<'py, PyArray2<PyFloat>>,
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
) -> DendrogramResult {
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
		let result = crate::cluster::hnsw_based_dendrogram::<PyFloat,usize,_,_>(
			&arrview2_py_to_rust(data.as_array()),
			graphidx::measures::SquaredEuclideanDistance::new(),
			min_pts,
			expand.unwrap_or(true),
			symmetric_expand.unwrap_or(false),
			hnsw_params,
		);
		#[cfg(feature="pyprec16")]
		let result = {
			let (part1, part2) = result;
			let part1 = part1.into_iter().map(|(a,b,d,size)| (a,b,d.max(PyFloat::ZERO).to_f32().sqrt(), size)).collect();
			let part2 = part2.into_iter().map(|d| d.max(PyFloat::ZERO).to_f32().sqrt()).collect();
			(part1, part2)
		};
		#[cfg(not(feature="pyprec16"))]
		let result = {
			let mut result = result;
			result.0.iter_mut().for_each(|(_,_,d,_)| *d = d.max(PyFloat::ZERO).sqrt());
			result.1.iter_mut().for_each(|d| *d = d.max(PyFloat::ZERO).sqrt());
			result
		};
		result
	}
}

#[pyfunction]
#[pyo3(signature = (data, min_pts, self_join_neighbors, query_max_heap_size, expand=None, symmetric_expand=None, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, query_local=false))]
pub fn hnsw_based_dendrogram_self_joined<'py>(
	data: Bound<'py, PyArray2<PyFloat>>,
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
) -> DendrogramResult {
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
		let result = crate::cluster::hnsw_based_dendrogram_self_joined::<PyFloat,usize,_,_>(
			&arrview2_py_to_rust(data.as_array()),
			graphidx::measures::SquaredEuclideanDistance::new(),
			min_pts,
			expand.unwrap_or(true),
			symmetric_expand.unwrap_or(false),
			hnsw_params,
			self_join_neighbors,
			query_max_heap_size,
			query_local.unwrap_or(false),
		);
		#[cfg(feature="pyprec16")]
		let result = {
			let (part1, part2) = result;
			let part1 = part1.into_iter().map(|(a,b,d,size)| (a,b,d.max(PyFloat::ZERO).to_f32().sqrt(), size)).collect();
			let part2 = part2.into_iter().map(|d| d.max(PyFloat::ZERO).to_f32().sqrt()).collect();
			(part1, part2)
		};
		#[cfg(not(feature="pyprec16"))]
		let result = {
			let mut result = result;
			result.0.iter_mut().for_each(|(_,_,d,_)| *d = d.max(PyFloat::ZERO).sqrt());
			result.1.iter_mut().for_each(|d| *d = d.max(PyFloat::ZERO).sqrt());
			result
		};
		result
	}
}




macro_rules! add_module_parts {
	($module: ident, classes = [$($name: ident),*$(,)*]$(,)?) => {
		$($module.add_class::<$name>()?;)*
	};
	($module: ident, functions = [$($name: ident),*$(,)*]$(,)?) => {
		$($module.add_function(wrap_pyfunction!($name, $module)?)?;)*
	};
	(
		$module: ident,
		$($label: ident = [$($name: ident),*$(,)?]),*
		$(,)?
	) => {
		$(add_module_parts!($module, $label = [$($name,)*]);)*
	};
}
#[pymodule(name="graphidxbaselines")]
fn graphidxbaselines(m: &Bound<'_, PyModule>) -> PyResult<()> {
	add_module_parts!(
		m,
		classes = [
			PyHNSW,PySENHNSW,SparsePyHNSW,
			PyFatHNSW,OwningPyHNSW,OwningPyFatHNSW,
			PyRNNDescent,PySENDescent,
			DistanceEnum,
		],
		functions = [
			load_hnswlib,load_hnswlib_fat,
			hnsw_based_dendrogram,hnsw_based_dendrogram_self_joined,
			ref_bits,prec_bits,
		]
	);
	add_distance_wrappers_to_module!(m);
	Ok(())
}


