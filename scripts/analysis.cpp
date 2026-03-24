#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Compute Pearson correlation matrix
MatrixXd correlation_matrix(const MatrixXd& X) {
	int n = X.cols();
	MatrixXd corr = MatrixXd::Zero(n, n);
	for (int i = 0; i < n; ++i) {
		for (int j = i; j < n; ++j) {
			double mean_i = X.col(i).mean();
			double mean_j = X.col(j).mean();
			double num = ((X.col(i).array() - mean_i) * (X.col(j).array() - mean_j)).sum();
			double denom = std::sqrt(((X.col(i).array() - mean_i).square().sum()) * ((X.col(j).array() - mean_j).square().sum()));
			double c = (denom == 0) ? 0 : num / denom;
			corr(i, j) = c;
			corr(j, i) = c;
		}
	}
	return corr;
}

// Compute distance matrix: sqrt(2 * (1 - corr))
MatrixXd distance_matrix(const MatrixXd& corr) {
	int n = corr.rows();
	MatrixXd dist = MatrixXd::Zero(n, n);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			dist(i, j) = std::sqrt(2.0 * (1.0 - corr(i, j)));
		}
	}
	return dist;
}

// Prim's algorithm for MST, returns edge list: (i, j, weight)
std::vector<std::tuple<int, int, double>> minimum_spanning_tree(const MatrixXd& dist) {
	int n = dist.rows();
	std::vector<bool> in_tree(n, false);
	std::vector<double> min_edge(n, std::numeric_limits<double>::max());
	std::vector<int> parent(n, -1);
	std::vector<std::tuple<int, int, double>> edges;
	min_edge[0] = 0;
	for (int i = 0; i < n; ++i) {
		int u = -1;
		for (int v = 0; v < n; ++v) {
			if (!in_tree[v] && (u == -1 || min_edge[v] < min_edge[u])) {
				u = v;
			}
		}
		in_tree[u] = true;
		if (parent[u] != -1) {
			edges.emplace_back(parent[u], u, dist(u, parent[u]));
		}
		for (int v = 0; v < n; ++v) {
			if (dist(u, v) < min_edge[v] && !in_tree[v]) {
				min_edge[v] = dist(u, v);
				parent[v] = u;
			}
		}
	}
	return edges;
}

// Build adjacency list from MST edge list
std::vector<std::vector<std::pair<int, double>>> build_adj_list(int n, const std::vector<std::tuple<int, int, double>>& edges) {
	std::vector<std::vector<std::pair<int, double>>> adj(n);
	for (const auto& e : edges) {
		int u, v; double w;
		std::tie(u, v, w) = e;
		adj[u].emplace_back(v, w);
		adj[v].emplace_back(u, w);
	}
	return adj;
}

// Compute ultrametric distance matrix from MST
MatrixXd ultrametric_matrix(int n, const std::vector<std::vector<std::pair<int, double>>>& adj) {
	MatrixXd U = MatrixXd::Zero(n, n);
	// For each pair, BFS to find max edge weight along path
	for (int i = 0; i < n; ++i) {
		std::vector<double> max_w(n, 0.0);
		std::vector<bool> visited(n, false);
		std::queue<int> q;
		q.push(i);
		visited[i] = true;
		while (!q.empty()) {
			int u = q.front(); q.pop();
			for (const auto& [v, w] : adj[u]) {
				if (!visited[v]) {
					max_w[v] = std::max(max_w[u], w);
					visited[v] = true;
					q.push(v);
				}
			}
		}
		for (int j = 0; j < n; ++j) {
			U(i, j) = max_w[j];
		}
	}
	return U;
}

// pybind11 wrapper
py::dict mst_and_ultrametric(py::array_t<double, py::array::c_style | py::array::forcecast> returns) {
	// returns: shape (n_samples, n_assets)
	py::buffer_info buf = returns.request();
	if (buf.ndim != 2) throw std::runtime_error("Input must be 2D array");
	int n_samples = buf.shape[0];
	int n_assets = buf.shape[1];
	MatrixXd X = Eigen::Map<MatrixXd>((double*)buf.ptr, n_samples, n_assets);
	MatrixXd corr = correlation_matrix(X);
	MatrixXd dist = distance_matrix(corr);
	auto edges = minimum_spanning_tree(dist);
	auto adj = build_adj_list(n_assets, edges);
	MatrixXd ultra = ultrametric_matrix(n_assets, adj);
	// Convert MST edge list to Python list
	std::vector<std::tuple<int, int, double>> mst_edges = edges;
	py::list py_edges;
	for (const auto& e : mst_edges) {
		py_edges.append(py::make_tuple(std::get<0>(e), std::get<1>(e), std::get<2>(e)));
	}
	return py::dict(
		"correlation"_a=corr,
		"distance"_a=dist,
		"mst_edges"_a=py_edges,
		"ultrametric"_a=ultra
	);
}

PYBIND11_MODULE(mst_cpp, m) {
	m.doc() = "MST and Ultrametric Distance (Eigen, pybind11)";
	m.def("mst_and_ultrametric", &mst_and_ultrametric, "Compute MST and ultrametric distance from returns matrix");
}
