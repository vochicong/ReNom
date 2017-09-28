let TopologyState = {
    loading: false,

    filename: "",
    algorithms: ["PCA", "TSNE", "Auto Encoder 2Layer", "Auto Encoder 3Layer", "Auto Encoder 4Layer"],
    algorithm_index: 0,
    algorithm_name: "PCA",
    labels: undefined,

    search_txt: "",

    resolution: 25,
    overlap: 0.5,
    color_index: 0,

    nodes: undefined,
    node_sizes: undefined,
    edges: undefined,
    colors: undefined,

    click_node_index: undefined,
    node_categorical_data: undefined,
    node_data: undefined,
    show_category: 0,
}

export default TopologyState