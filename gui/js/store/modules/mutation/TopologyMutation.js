let TopologyMutation = {
    reset_all(state) {
        state.filename = "";
        state.algorithm_index = 0;
        state.algorithm_name = "PCA";
        state.labels = undefined;
        state.search_txt = "";
        state.resolution = 25;
        state.overlap = 0.5;
        state.color_index = 0;
        state.nodes = undefined;
        state.node_sizes = undefined;
        state.edges = undefined;
        state.colors = undefined;
        state.click_node_index = undefined;
        state.node_categorical_data = undefined;
        state.node_data = undefined;
        state.show_category = 0;
    },
    reset_topology(state) {
        state.nodes = undefined;
        state.node_sizes = undefined;
        state.edges = undefined;
        state.colors = undefined;
        state.click_node_index = undefined;
        state.node_categorical_data = undefined;
        state.node_data = undefined;
        state.show_category = 0;
    },
    set_loading(state, payload) {
        state.loading = payload.loading;
    },
    set_filename(state, payload) {
        state.filename = payload.filename;
    },
    set_algorithm(state, payload) {
        state.algorithm_index = payload.index;
        state.algorithm_name = state.algorithms[payload.index];
    },
    set_labels(state, payload) {
        state.labels = payload.labels;
    },
    set_params(state, payload) {
        state.resolution = payload.resolution;
        state.overlap = payload.overlap;
        state.color_index = payload.color_index;
    },
    set_topology_result(state, payload) {
        state.nodes = payload.nodes;
        state.node_sizes = payload.node_sizes;
        state.edges = payload.edges;
        state.colors = payload.colors;
    },
    set_click_node(state, payload) {
        state.click_node_index = payload.click_node_index;
        state.node_categorical_data = payload.node_categorical_data;
        state.node_data = payload.node_data;
    },
    set_search_txt(state, payload) {
        state.search_txt = payload.search_txt;
    }
}

export default TopologyMutation;