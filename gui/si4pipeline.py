
import pickle
import traceback

import numpy as np
import pandas as pd
import si4automl as plp
import streamlit as st

from barfi import Block, barfi_schemas, st_barfi

load = Block(name='Dataset Loader')
load.add_output(name='X')
load.add_output(name='y')


def load_func(self):
    st.session_state["executed"] = True
    if 'results_df' in st.session_state:
        del st.session_state['results_df']

    print('load')
    X, y = plp.initialize_dataset()
    plp_setting = {}
    plp_setting["tune_flag"] = False
    self.set_interface(name='X', value=(plp_setting, X))
    self.set_interface(name='y', value=(plp_setting, y))


load.add_compute(load_func)


def get_parameters(parameter_text, type_converter=float):
    if "," in parameter_text:
        tune_flag = True
        return tune_flag, [type_converter(each) for each in parameter_text.split(",")]
    else:
        tune_flag = False
        return tune_flag, type_converter(parameter_text)


def create_node(node_name, node_func, outputs=["O"], inputs=["X","y"], options={}):

    for i, each_output in enumerate(outputs):
        if each_output in inputs:
            outputs[i] = each_output + "'"

    node = Block(name=node_name)

    for each_input in inputs:
        node.add_input(name=each_input)

    for each_output in outputs:
        node.add_output(name=each_output)

    for each_option in options:
        node.add_option(name='text', type="display", value=each_option)
        node.add_option(
            name=each_option, type="input", value=str(options[each_option]["default"])
        )

    def compute_func(self):
        print(f"---{node_name}---")
        input_variables = []

        for each_input in self._inputs:
            (plp_setting, X) = self.get_interface(name=each_input)
            input_variables.append(X)

        assert len(options)<=1
        print(f"{node_func=}")
        print(f"{input_variables=}")
        print(f"{options=}")
        if len(options) == 0:
            output_variables = node_func(*input_variables)
        elif len(options) == 1:
            option = next(iter(options))
            #print(self.get_option(name=option))
            type_converter = options[option]["type"]
            tune_flag, parameters = get_parameters(
                self.get_option(name=option), type_converter
            )
            # parameters[option] = parameters_
            print("parameters", parameters)
            print("tune_flag", tune_flag)

            if not tune_flag:
                input_variables.append(parameters)
                output_variables = node_func(*input_variables)
            else:
                plp_setting["tune_flag"] = tune_flag
                output_variables = node_func(*input_variables, parameters=parameters)
        print(f"{output_variables=}")
        if len(outputs) == 1:
            self.set_interface(name=outputs[0], value=(plp_setting, output_variables))
        elif len(outputs) == 2:
            for i, each_output in enumerate(outputs):
                self.set_interface(
                    name=each_output, value=(plp_setting, output_variables[i])
                )
        else:
            raise Warning('the number of outputs should be 1 or 2')
        print(outputs)
        print("----------")

    node.add_compute(compute_func)
    return node


mean_value_imputation = create_node(
    "mean_value_imputation",
    plp.mean_value_imputation,
    outputs=["y'"],
    inputs=["X", "y"],
)
soft_ipod = create_node(
    "soft_ipod",
    plp.soft_ipod,
    outputs=["O"],
    inputs=["X", "y"],
    options={"penalty coefficient": {"default": 0.015, "type": float}},
)
remove_outliers = create_node(
    "remove_outliers", plp.remove_outliers, outputs=["X", "y"], inputs=["X", "y", "O"]
)
marginal_screening = create_node(
    "marginal_screening",
    plp.marginal_screening,
    outputs=["M"],
    inputs=["X", "y"],
    options={"number of features": {"default": 5, "type": int}},
)
extract_features = create_node(
    "extract_features", plp.extract_features, outputs=["X"], inputs=["X", "M"]
)
stepwise_feature_selection = create_node(
    "stepwise_feature_selection",
    plp.stepwise_feature_selection,
    outputs=["M"],
    inputs=["X", "y"],
    options={"number of features": {"default": 3, "type": int}},
)
lasso = create_node(
    "lasso",
    plp.lasso,
    outputs=["M"],
    inputs=["X", "y"],
    options={"penalty coefficient": {"default": 0.08, "type": float}},
)
union = create_node("union", plp.union, outputs=["M"], inputs=["M1", "M2"])
regression_imputation = create_node(
    "regression_imputation",
    plp.definite_regression_imputation,
    outputs=["y'"],
    inputs=["X", "y"],
)
cook_distance = create_node(
    "cook_distance",
    plp.cook_distance,
    outputs=["O"],
    inputs=["X", "y"],
    options={"penalty coefficient": {"default": 3.0, "type": float}},
)
intersection = create_node(
    "intersection", plp.intersection, outputs=["M"], inputs=["M1", "M2"]
)


class BaseTest:
    def perform_inference(self2, self):
        print('perform inference')
        
        plp_setting, pipeline = self2.make_pipeline(self)
        try:
            if st.session_state.dataset == 'random':
                n, p = 100, 10
                rng = np.random.default_rng(0)
                X = rng.normal(size=(n, p))
                y = rng.normal(size=n)
                num_missing = rng.binomial(n, 0.03)
                mask = rng.choice(n, num_missing, replace=False)
                y[mask] = np.nan
                sigma = 1.0

                if plp_setting["tune_flag"]:
                    pipeline.tune(X,
                                  y, 
                                  num_folds=st.session_state.cv,
                                  random_state=0)

                M, p_list = pipeline.inference(X, y, sigma)
                for each_feature, p_value in zip(M, p_list):
                    print(f'feature:{each_feature} p-value:{p_value:.3f}')
                results = []
                for each_feature, p_value in zip(M, p_list):
                    significance_status = "significant" if p_value <= 0.05 else "not significant"
                    result = {
                        'Feature': f"feature_{each_feature}",
                        'p-value': round(p_value, 3),
                        'Significance': significance_status
                    }
                    results.append(result)
                results_df = pd.DataFrame(results)
                st.session_state['results_df'] = results_df

            elif st.session_state.dataset in [
                "prostate_cancer",
                "red_wine",
                "concrete",
                "abalone",
                "uploaded",
            ]:
                if st.session_state.dataset == "uploaded":
                    X, y, features = st.session_state.uploaded_dataset

                elif st.session_state.dataset == 'prostate_cancer':
                    features = [
                        'lcavol', 'lweight', 'age', 'lbph', 'svi', 
                        'lcp', 'gleason', 'pgg45'
                    ]
                    with open("dataset/prostate_cancer.pkl", "rb") as f:
                        X, y = pickle.load(f)

                elif st.session_state.dataset == 'red_wine':
                    features = [
                        "fixed_acidity", "volatile_acidity", "citric_acid",
                        "residual_sugar", "chlorides", "free_sulfur_dioxide",
                        "total_sulfur_dioxide", "density", "pH",
                        "sulphates", "alcohol"
                    ]
                    with open("dataset/red_wine.pkl", "rb") as f:
                        X, y = pickle.load(f)

                elif st.session_state.dataset == 'concrete':
                    features = [
                        "cement", "blast_furnace_slag", "fly_ash",
                        "water", "superplasticizer", "coarse_aggregate",
                        "fine_aggregate", "age"
                    ]
                    with open("dataset/concrete.pkl", "rb") as f:
                        X, y = pickle.load(f)   

                elif st.session_state.dataset == 'abalone':
                    features = [
                        "length", "diameter",
                        "height", "whole_weight", "shucked_weight",
                        "viscera_weight", "shell_weight"
                    ]
                    with open("dataset/abalone.pkl", "rb") as f:
                        X, y = pickle.load(f)

                if plp_setting["tune_flag"]:
                    pipeline.tune(X,
                                  y, 
                                  num_folds=st.session_state.cv,
                                  random_state=0)

                M, p_list = pipeline.inference(X, y)

                print("Inference results are :\n")
                for each_feature, p_value in zip(M, p_list):
                    print(
                        f'{features[each_feature]}:\np-value is {p_value:.6f}, \
                            {"significant" if p_value <= 0.05 else "not significant"}\n'
                    )
                    significance_status = "significant"\
                        if p_value < 0.05 else "not significant"
                results = []
                for each_feature, p_value in zip(M, p_list):
                    significance_status = "significant" if p_value <= 0.05 else "not significant"
                    result = {
                        'Feature': features[each_feature],
                        'p-value': round(p_value, 6),
                        'Significance': significance_status
                    }
                    results.append(result)
                results_df = pd.DataFrame(results)
                st.session_state['results_df'] = results_df
            else:
                raise Warning('unknown dataset')
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    def make_pipeline(self2, self):
        raise NotImplementedError("make_pipeline method is not implemented")


class Test(BaseTest):
    def make_pipeline(self2, self):
        (plp_setting, M) = self.get_interface(name='M')
        pipeline = plp.construct_pipelines(output=M)
        return plp_setting, pipeline


class TestWithMultiPipelines(BaseTest):
    def make_pipeline(self2, self):
        (plp_setting, M1) = self.get_interface(name='M1')
        (plp_setting, M2) = self.get_interface(name='M2')
        manager_op1_mul = plp.construct_pipelines(output=M1)
        manager_op2_mul = plp.construct_pipelines(output=M2)
        manager = manager_op1_mul | manager_op2_mul

        plp_setting["tune_flag"] = True
        return plp_setting, manager


test = Block(name='test')
test.add_input(name='M')
test.add_compute(Test().perform_inference)


test_with_multi_pipeline = Block(name='test_with_multi_pipeline')
test_with_multi_pipeline.add_input(name='M1')
test_with_multi_pipeline.add_input(name='M2')
test_with_multi_pipeline.add_compute(TestWithMultiPipelines().perform_inference)

base_blocks = [
    load,
    test,
    test_with_multi_pipeline,
    mean_value_imputation,
    soft_ipod,
    remove_outliers,
    marginal_screening,
    extract_features,
    stepwise_feature_selection,
    lasso,
    union,
    regression_imputation,
    cook_distance,
    intersection,
]


def main():
    st.set_page_config(
        page_icon="🍍",
        page_title="SI4PIPELINE",
        layout="wide")

    st.title('SI4PIPELINE')

    st.sidebar.title('Settings')
    cv = st.sidebar.slider('Number of folds in cross-validation:', 0, 10, 5)
    if cv not in st.session_state:
        st.session_state.cv = cv

    # load data
    st.header("STEP1: Upload data")
    _, col1, col2 = st.columns([1, 8, 7])
    with col1:
        uploaded_file = st.file_uploader("Upload your own data", type="csv")
        if uploaded_file is not None:
            header_exists = st.checkbox('The file have a header', value=True)
            if header_exists:
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file, header=None)
            default_target_column = data.columns[-1]
            target_column = st.text_input('Target column name:', value=default_target_column)
            y = data[target_column].values
            X = data.drop(columns=[target_column]).values
            features = data.drop(columns=[target_column]).columns
            st.session_state.dataset = 'uploaded'
            st.session_state.uploaded_dataset = [X, y, features]
    with col2:
        # if st.checkbox('or select existing dataset'):
        if uploaded_file is None:
            existing_data_options = [
                "-",
                "prostate_cancer",
                "random",
                "red_wine",
                "concrete",
                "abalone",
            ] 
            # デモ用のデータセットを選択
            selected_dataset = st.selectbox(
                "Or select a demo dataset:", existing_data_options
            )
            if selected_dataset != "-":
                st.session_state.dataset = selected_dataset
    if uploaded_file:
        _, col1 = st.columns([1, 15])
        with col1:
            with st.expander('Show data'):
                st.dataframe(data, height=300)

    # load and define pipeline
    st.header('STEP2: Define and execute pipeline')
    _, col1, col2 = st.columns([1, 8, 7])
    with col1:
        st.write('Define your data processing pipeline')
        # st.write('You can create blocks by right-clicking and connect them to create a pipeline.')
        # st.write('You can also set parameters for each block.')
        # st.write('After defining the pipeline, click the "Execute" button to perform the analysis.')
        # st.write('The results will be displayed in the next section.')
    with col2:
        load_pipeline = st.selectbox(
            "Or select a pre-defined pipeline:", barfi_schemas()
        )

    _, col1 = st.columns([1, 15])

    with col1:
        barfi_result = st_barfi(
            base_blocks=base_blocks, compute_engine=True, load_schema=load_pipeline
        )

    # inference results
    st.header('STEP3: Inference results')
    _, col1 = st.columns([1, 15])
    with col1:
        if 'results_df' not in st.session_state:
            if 'executed' in st.session_state:
                st.error("The analysis has failed.\n\
                         Please check the pipeline structure and the dataset format.")
            else:
                st.write("(No analysis has been performed yet.)")
        else:
            def highlight_significant(row):
                color = 'background-color: green' if row['Significance'] == 'significant' else ''
                return [color] * len(row)
            styled_df = st.session_state['results_df'].style.apply(highlight_significant, axis=1)
            st.dataframe(styled_df)



if __name__ == '__main__':
    main()
