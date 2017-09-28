import axios from 'axios'

let TopologyAction = {
    load_file(context, payload) {
        context.commit('set_loading', {
            'loading': true
        });
        context.commit('reset_all');
        let e = document.getElementById("searchtxtbox");
        e.value = "";

        let fd = new FormData();
        fd.append("filename", payload.filename);
        fd.append("algorithm", payload.algorithm_index);

        return axios.post('/api/load', fd)
            .then(function(response){
                let error = response.data.error;
                if (error){
                    alert("File not found. Please try again.");
                    context.commit('set_loading', {
                        'loading': false
                    });
                    return
                }

                context.commit('set_filename', {
                    'filename': payload.filename
                });
                context.commit('set_algorithm', {
                    'index': payload.algorithm_index
                });
                context.commit('set_labels', {
                    'labels': response.data.labels
                });
                context.commit('set_loading', {
                    'loading': false
                });
            });
    },
    create_topology(context, payload) {
        context.commit('set_loading', {
            'loading': true
        });
        context.commit('reset_topology');

        let fd = new FormData();
        fd.append("filename", context.state.filename);
        fd.append("algorithm", context.state.algorithm_index);
        fd.append("resolution", payload.resolution);
        fd.append("overlap", payload.overlap);
        fd.append("colored_by", payload.color_index);
        fd.append("search_value", context.state.search_txt);

        return axios.post('/api/create', fd)
            .then(function(response){
                context.commit('set_params', {
                    'resolution': payload.resolution,
                    'overlap': payload.overlap,
                    'color_index': payload.color_index
                });

                context.commit('set_topology_result', {
                    'nodes': response.data.nodes,
                    'node_sizes': response.data.node_sizes,
                    'edges': response.data.edges,
                    'colors': response.data.colors
                });

                context.commit('set_loading', {
                    'loading': false
                });
            });
    },
    search_topology(context, payload) {
        context.commit('set_loading', {
            'loading': true
        });

        let fd = new FormData();
        fd.append("filename", context.state.filename);
        fd.append("algorithm", context.state.algorithm_index);
        fd.append("resolution", context.state.resolution);
        fd.append("overlap", context.state.overlap);
        fd.append("colored_by", context.state.color_index);
        fd.append("search_value", payload.search_txt);

        return axios.post('/api/create', fd)
            .then(function(response){
                context.commit('set_search_txt', {
                    'search_txt': payload.search_txt
                });

                context.commit('set_topology_result', {
                    'nodes': response.data.nodes,
                    'node_sizes': response.data.node_sizes,
                    'edges': response.data.edges,
                    'colors': response.data.colors
                });

                context.commit('set_loading', {
                    'loading': false
                });
            });
    },
    click_node(context, payload) {
        if (context.state.click_node_index) {
            let circle = d3.select("#circle_"+context.state.click_node_index);
            circle.attr("stroke","rgba(153,153,153,0)");
        }

        let fd = new FormData();
        fd.append("clicknode", payload.click_node_index);

        return axios.post('/api/click', fd)
        .then(function(response){
            context.commit('set_click_node', {
                'click_node_index': payload.click_node_index,
                'node_categorical_data': response.data.categorical_data,
                'node_data': response.data.data
            })

            let circle = d3.select("#circle_"+payload.click_node_index);
            circle.attr("stroke","rgba(153,153,153,0.3)").attr("stroke-width",5);
        });
    }
}

export default TopologyAction;