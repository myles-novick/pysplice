from sys import stderr
from typing import List, Dict, Optional, Union, Callable
from datetime import datetime, date
from inspect import getsource

import pandas as pd
from pandas import DataFrame as PandasDF
import cloudpickle
import base64

from pyspark.sql.dataframe import DataFrame as SparkDF
from pyspark.ml import Pipeline as MLPipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler

from splicemachine import SpliceMachineException
from splicemachine.features.utils.feature_utils import sql_to_datatype
from splicemachine.spark import PySpliceContext, ExtPySpliceContext
from splicemachine.features import Feature, FeatureSet
from .training_set import TrainingSet
from .utils import display
from .utils.training_utils import ReturnType, _format_training_set_output
from .utils.http_utils import RequestType, make_request, _get_feature_store_url, Endpoints, _get_credentials, _get_token
from .pipe import Pipe
from .pipeline import Pipeline

from .constants import PipeLanguage, SQL, FeatureType, PipeType
from .training_view import TrainingView
import warnings
from requests.auth import HTTPBasicAuth


class FeatureStore:
    def __init__(self, splice_ctx: PySpliceContext = None) -> None:
        self.splice_ctx = splice_ctx
        self.mlflow_ctx = None
        self.feature_sets = []  # Cache of newly created feature sets
        self._FS_URL = _get_feature_store_url()
        if not self._FS_URL: warnings.warn(
            "Uh Oh! FS_URL variable was not found... you should call 'fs.set_feature_store_url(<url>)' before doing anything.")
        self._auth = None
        self.__try_auto_login()

    def register_splice_context(self, splice_ctx: PySpliceContext) -> None:
        if not (isinstance(splice_ctx, PySpliceContext) or isinstance(splice_ctx, ExtPySpliceContext)):
            raise SpliceMachineException(f'Splice Context must be of type PySpliceContext or ExtPySpliceContext but is'
                                         f'of type {type(splice_ctx)}')
        self.splice_ctx = splice_ctx

    def get_feature_sets(self, feature_set_names: List[str] = None) -> List[FeatureSet]:
        """
        Returns a list of available feature sets
        
        :param feature_set_names: A list of feature set names in the format '{schema_name}.{table_name}'. If none will return all FeatureSets
        :return: List[FeatureSet] the list of Feature Sets
        """

        r = make_request(self._FS_URL, Endpoints.FEATURE_SETS, RequestType.GET, self._auth,
                         {"name": feature_set_names} if feature_set_names else None)
        return [FeatureSet(**fs) for fs in r]

    def remove_training_view(self, name: str, version: Union[str, int] = 'latest'):
        """
        This removes a training view if it is not being used by any currently deployed models.
        NOTE: Once this training view is removed, you will not be able to deploy any models that were trained using this
        view

        :param name: The view name
        :param version: The view version
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        print(f"Removing Training View {name}...", end=' ')
        make_request(self._FS_URL, f'{Endpoints.TRAINING_VIEWS}/{name}', RequestType.DELETE, self._auth,
                     params={'version': version})
        print('Done.')

    def get_summary(self) -> Dict[str, str]:
        """
        This function returns a summary of the feature store including:\n
        * Number of feature sets
        * Number of deployed feature sets
        * Number of features
        * Number of deployed features
        * Number of training sets
        * Number of training views
        * Number of associated models - this is a count of the MLManager.RUNS table where the `splice.model_name` tag is set and the `splice.feature_store.training_set` parameter is set
        * Number of active (deployed) models (that have used the feature store for training)
        * Number of pending feature sets - this will will require a new table `featurestore.pending_feature_set_deployments` and it will be a count of that
        """

        r = make_request(self._FS_URL, Endpoints.SUMMARY, RequestType.GET, self._auth)
        return r

    def get_training_view(self, training_view: str, version: Union[int, str] = 'latest') -> TrainingView:
        """
        Gets a training view by name

        :param training_view: Training view name
        :param version: Training view version
        :return: TrainingView
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        r = make_request(self._FS_URL, f'{Endpoints.TRAINING_VIEWS}/{training_view}', RequestType.GET, self._auth,
                         params={'version': version})
        return TrainingView(**r[0])

    def get_training_views(self, _filter: Dict[str, Union[int, str]] = None) -> List[TrainingView]:
        """
        Returns a list of all available training views with an optional filter

        :param _filter: Dictionary container the filter keyword (label, description etc) and the value to filter on
            If None, will return all TrainingViews
        :return: List[TrainingView]
        """

        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEWS, RequestType.GET, self._auth)
        return [TrainingView(**tv) for tv in r]

    def get_training_view_id(self, name: str) -> int:
        """
        Returns the unique view ID from a name

        :param name: The training view name
        :return: The training view id
        """
        # return self.splice_ctx.df(SQL.get_training_view_id.format(name=name)).collect()[0][0]
        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_ID, RequestType.GET, self._auth, {"name": name})
        return int(r)

    def get_features_by_name(self, names: Optional[List[str]] = None, as_list=False) -> Union[List[Feature], PandasDF]:
        """
        Returns a dataframe or list of features whose names are provided

        :param names: The list of feature names
        :param as_list: Whether or not to return a list of features. Default False
        :return: SparkDF or List[Feature] The list of Feature objects or Spark Dataframe of features and their metadata. Note, this is not the Feature
        values, simply the describing metadata about the features. To create a training dataset with Feature values, see
        :py:meth:`features.FeatureStore.get_training_set` or :py:meth:`features.FeatureStore.get_feature_dataset`
        """
        r = make_request(self._FS_URL, Endpoints.FEATURES, RequestType.GET, self._auth, {"name": names})
        return [Feature(**f) for f in r] if as_list else pd.DataFrame.from_dict(r)

    def training_view_exists(self, name: str) -> bool:
        """
        Returns if a training view exists or not

        :param name: The training view name
        :return: bool True if the training view exists, False otherwise
        """
        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_EXISTS, RequestType.GET, self._auth,
                         params={"name": name})
        return r

    def feature_exists(self, name: str) -> bool:
        """
        Returns if a feature exists or not

        :param name: The feature name
        :return: bool True if the feature exists, False otherwise
        """
        r = make_request(self._FS_URL, Endpoints.FEATURE_EXISTS, RequestType.GET, self._auth, params={"name": name})
        return r

    def feature_set_exists(self, schema: str, table: str) -> bool:
        """
        Returns if a feature set exists or not

        :param schema: The feature set schema
        :param table: The feature set table
        :return: bool True if the feature exists, False otherwise
        """
        r = make_request(self._FS_URL, Endpoints.FEATURE_SET_EXISTS, RequestType.GET, self._auth,
                         params={"schema": schema, "table": table})
        return r

    def get_feature_details(self, name: str) -> Feature:
        """
        Returns a Feature and it's detailed information

        :param name: The feature name
        :return: Feature
        """
        r = make_request(self._FS_URL, Endpoints.FEATURE_DETAILS, RequestType.GET, self._auth, {"name": name})
        return Feature(**r)

    def get_feature_vector(self, features: List[Union[str, Feature]],
                           join_key_values: Dict[str, str], return_primary_keys=True, return_sql=False) -> Union[
        str, PandasDF]:
        """
        Gets a feature vector given a list of Features and primary key values for their corresponding Feature Sets

        :param features: List of str Feature names or Features
        :param join_key_values: (dict) join key values to get the proper Feature values formatted as {join_key_column_name: join_key_value}
        :param return_primary_keys: Whether to return the Feature Set primary keys in the vector. Default True
        :param return_sql: Whether to return the SQL needed to get the vector or the values themselves. Default False
        :return: Pandas Dataframe or str (SQL statement)
        """
        features = [f if isinstance(f, str) else f.__dict__ for f in features]
        r = make_request(self._FS_URL, Endpoints.FEATURE_VECTOR, RequestType.POST, self._auth,
                         params={"pks": return_primary_keys, "sql": return_sql},
                         body={"features": features, "join_key_values": join_key_values})
        return r if return_sql else pd.DataFrame(r, index=[0])

    def get_feature_vector_sql_from_training_view(self, training_view: str, features: List[Union[str, Feature]]) -> str:
        """
        Returns the parameterized feature retrieval SQL used for online model serving.

        :param training_view: (str) The name of the registered training view
        :param features: (List[str]) the list of features from the feature store to be included in the training

            :NOTE:
                .. code-block:: text

                    This function will error if the view SQL is missing a view key required \n
                    to retrieve the desired features

        :return: (str) the parameterized feature vector SQL
        """
        features = [f if isinstance(f, str) else f.__dict__ for f in features]
        r = make_request(self._FS_URL, Endpoints.FEATURE_VECTOR_SQL, RequestType.POST, self._auth,
                         {"view": training_view}, features)
        return r

    def get_feature_primary_keys(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Returns a dictionary mapping each individual feature to its primary key(s). This function is not yet implemented.

        :param features: (List[str]) The list of features to get primary keys for
        :return: Dict[str, List[str]] A mapping of {feature name: [pk1, pk2, etc]}
        """
        pass

    def get_training_view_features(self, training_view: str, version: Union[int, str] = 'latest') -> List[Feature]:
        """
        Returns the available features for the given a training view name

        :param training_view: The name of the training view
        :param version: The version of the training view
        :return: A list of available Feature objects
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_FEATURES, RequestType.GET,
                         self._auth, {"view": training_view, 'version': version})
        return [Feature(**f) for f in r]

    def get_feature_description(self):
        # TODO
        raise NotImplementedError

    def get_training_set(self, features: Union[List[Feature], List[str]], current_values_only: bool = False,
                         start_time: datetime = None, end_time: datetime = None, label: str = None,
                         return_pk_cols: bool = False,
                         return_ts_col: bool = False, return_type: str = 'spark',
                         return_sql: bool = False, save_as: str = None) -> SparkDF or str:
        """
        Gets a set of feature values across feature sets that is not time dependent (ie for non time series clustering).
        This feature dataset will be treated and tracked implicitly the same way a training_dataset is tracked from
        :py:meth:`features.FeatureStore.get_training_set` . The dataset's metadata and features used will be tracked in mlflow automatically (see
        get_training_set for more details).

        :NOTE:
            .. code-block:: text

                The way point-in-time correctness is guaranteed here is by choosing one of the Feature Sets as the "anchor" dataset.
                This means that the points in time that the query is based off of will be the points in time in which the anchor
                Feature Set recorded changes. The anchor Feature Set is the Feature Set that contains the superset of all primary key
                columns across all Feature Sets from all Features provided. If more than 1 Feature Set has the superset of
                all Feature Sets, the Feature Set with the most primary keys is selected. If more than 1 Feature Set has the same
                maximum number of primary keys, the Feature Set is chosen by alphabetical order (schema_name, table_name).

        :param features: List of Features or strings of feature names

            :NOTE:
                .. code-block:: text

                    The Features Sets which the list of Features come from must have common join keys,
                    otherwise the function will fail. If there is no common join key, it is recommended to
                    create a Training View to specify the join conditions.

        :param current_values_only: If you only want the most recent values of the features, set this to true. Otherwise, all history will be returned. Default False
        :param start_time: How far back in history you want Feature values. If not specified (and current_values_only is False), all history will be returned.
            This parameter only takes effect if current_values_only is False.
        :param end_time: The most recent values for each selected Feature. This will be the cutoff time, such that any Feature values that
            were updated after this point in time won't be selected. If not specified (and current_values_only is False),
            Feature values up to the moment in time you call the function (now) will be retrieved. This parameter
            only takes effect if current_values_only is False.
        :param label: An optional label to specify for the training set. If specified, the feature set of that feature
            will be used as the "anchor" feature set, meaning all point in time joins will be made to the timestamps of
            that feature set. This feature will also be recorded as a "label" feature for this particular training set
            (but not others in the future, unless this label is again specified).
        :param return_pk_cols: bool Whether or not the returned sql should include the primary key column(s)
        :param return_ts_cols: bool Whether or not the returned sql should include the timestamp column
        :param return_type: How the data should be returned. If not specified, a Spark DF will be returned.
            Available arguments are: 'spark', 'pandas', 'json', 'sql'
            sql will return the SQL necessary to generate the dataframe
        :param save_as: Whether or not to save this Training Set (metadata) in the feature store for reproducibility. This
            enables you to version and persist the metadata for a training set of a specific model development. If you are
            using the Splice Machine managed MLFlow Service, this will be fully automated and managed for you upon model deployment,
            however you can still use this parameter to customize the name of the training set (it will default to the run id).
            If you are NOT using Splice Machine's mlflow service, this is a useful way to link specific modeling experiments
            to the exact training sets used. This DOES NOT persist the training set itself, rather the metadata required
            to reproduce the identical training set.
        :return: Spark DF or SQL statement necessary to generate the Training Set
        """

        # ~ Backwards Compatability ~
        if return_sql:
            print("Deprecated Parameter 'return_sql'. Use return_type='sql' ", file=stderr)
            return_type = 'sql'

        if return_type not in ReturnType.get_valid():
            raise SpliceMachineException(f'Return type must be one of {ReturnType.get_valid()}')

        features = [f if isinstance(f, str) else f.__dict__ for f in features]
        r = make_request(self._FS_URL, Endpoints.TRAINING_SETS, RequestType.POST, self._auth,
                         params={"current": current_values_only, "label": label, "pks": return_pk_cols,
                                 "ts": return_ts_col,
                                 'save_as': save_as, 'return_type': ReturnType.map_to_request(return_type)},
                         body={"features": features, "start_time": start_time, "end_time": end_time})
        create_time = r['metadata']['training_set_create_ts']
        start_time = r['metadata']['training_set_start_ts']
        end_time = r['metadata']['training_set_end_ts']
        sql = r['sql']
        tvw = TrainingView(**r['training_view']) if r.get('training_view') else None
        features = [Feature(**f) for f in r['features']]

        if self.mlflow_ctx and return_type != 'sql':
            # These will only exist if the user called "save_as" otherwise they will be None
            training_set_id = r['metadata'].get('training_set_id')
            training_set_version = r['metadata'].get('training_set_version')
            self.link_training_set_to_mlflow(features, create_time, start_time, end_time, tvw,
                                             training_set_id=training_set_id,
                                             training_set_version=training_set_version, training_set_name=save_as)

        return _format_training_set_output(response=r, return_type=return_type, splice_ctx=self.splice_ctx)

    def get_training_set_by_name(self, name, version: int = None, return_pk_cols: bool = False,
                                 return_ts_col: bool = False, return_sql=False, return_type: str = 'spark'):
        """
        Returns a Spark DF (or SQL) of an EXISTING Training Set (one that was saved with the save_as parameter in
        :py:meth:`~fs.get_training_set` or :py:meth:`~fs.get_training_set_from_view`. This is useful if you've deployed
        a model with a Training Set and

        :param name: Training Set name
        :param version: The version of this training set. If not set, it will grab the newest version
        :param return_pk_cols: bool Whether or not the returned sql should include the primary key column(s)
        :param return_ts_cols: bool Whether or not the returned sql should include the timestamp column
        :param return_sql: [DEPRECATED] (Optional[bool]) Return the SQL statement (str) instead of the Spark DF. Defaults False
        :param return_type: How the data should be returned. If not specified, a Spark DF will be returned.
            Available arguments are: 'spark', 'pandas', 'json', 'sql'
            sql will return the SQL necessary to generate the dataframe
        :return: Spark DF or SQL
        """

        # ~ Backwards Compatability ~
        if return_sql:
            print("Deprecated Parameter 'return_sql'. Use return_type='sql' ", file=stderr)
            return_type = 'sql'

        if return_type not in ReturnType.get_valid():
            raise SpliceMachineException(f'Return type must be one of {ReturnType.get_valid()}')

        r = make_request(self._FS_URL, f'{Endpoints.TRAINING_SETS}/{name}', RequestType.GET, self._auth,
                         params={"version": version, "pks": return_pk_cols, "ts": return_ts_col,
                                 'return_type': ReturnType.map_to_request(return_type)})
        sql = r["sql"]
        tvw = TrainingView(**r["training_view"])
        features = [Feature(**f) for f in r["features"]]
        create_time = r['metadata']['training_set_create_ts']
        start_time = r['metadata']['training_set_start_ts']
        end_time = r['metadata']['training_set_end_ts']
        # Link this to mlflow for reproducibility and model deployment
        if self.mlflow_ctx and not return_sql:
            # These will only exist if the user called "save_as" otherwise they will be None
            training_set_id = r['metadata'].get('training_set_id')
            self.link_training_set_to_mlflow(features, create_time, start_time, end_time, tvw,
                                             training_set_id=training_set_id,
                                             training_set_version=version, training_set_name=name)

        return _format_training_set_output(response=r, return_type=return_type, splice_ctx=self.splice_ctx)

    def get_training_set_from_view(self, training_view: str, features: Union[List[Feature], List[str]] = None,
                                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                                   return_pk_cols: bool = False, return_ts_col: bool = False, return_sql: bool = False,
                                   return_type: str = 'spark', save_as: str = None) -> SparkDF or str:
        """
        Returns the training set as a Spark Dataframe from a Training View. When a user calls this function (assuming they have registered
        the feature store with mlflow using :py:meth:`~mlflow.register_feature_store` )
        the training dataset's metadata will be tracked in mlflow automatically.\n
        The following will be tracked:\n
        * Training View
        * Selected features
        * Start time
        * End time

        This tracking will occur in the current run (if there is an active run)
        or in the next run that is started after calling this function (if no run is currently active).

        :param training_view: (str) The name of the registered training view
        :param features: (List[str] OR List[Feature]) the list of features from the feature store to be included in the training.
            If a list of strings is passed in it will be converted to a list of Feature. If not provided will return all available features.

            :NOTE:
                .. code-block:: text

                    This function will error if the view SQL is missing a join key required to retrieve the
                    desired features

        :param start_time: (Optional[datetime]) The start time of the query (how far back in the data to start). Default None

            :NOTE:
                .. code-block:: text

                    If start_time is None, query will start from beginning of history

        :param end_time: (Optional[datetime]) The end time of the query (how far recent in the data to get). Default None

            :NOTE:
                .. code-block:: text

                    If end_time is None, query will get most recently available data

        :param return_pk_cols: bool Whether or not the returned sql should include the primary key column(s)
        :param return_ts_cols: bool Whether or not the returned sql should include the timestamp column
        :param return_sql: [DEPRECATED] (Optional[bool]) Return the SQL statement (str) instead of the Spark DF. Defaults False
        :param return_type: How the data should be returned. If not specified, a Spark DF will be returned.
            Available arguments are: 'spark', 'pandas', 'json', 'sql'
            sql will return the SQL necessary to generate the dataframe
        :param save_as: Whether or not to save this Training Set (metadata) in the feature store for reproducibility. This
            enables you to version and persist the metadata for a training set of a specific model development. If you are
            using the Splice Machine managed MLFlow Service, this will be fully automated and managed for you upon model deployment,
            however you can still use this parameter to customize the name of the training set (it will default to the run id).
            If you are NOT using Splice Machine's mlflow service, this is a useful way to link specific modeling experiments
            to the exact training sets used. This DOES NOT persist the training set itself, rather the metadata required
            to reproduce the identical training set.
        :return: Optional[SparkDF, str] The Spark dataframe of the training set or the SQL that is used to generate it (for debugging)
        """

        # ~ Backwards Compatability ~
        if return_sql:
            print("Deprecated Parameter 'return_sql'. Use return_type='sql' ", file=stderr)
            return_type = 'sql'

        if return_type not in ReturnType.get_valid():
            raise SpliceMachineException(f'Return type must be one of {ReturnType.get_valid()}')

        # Generate the SQL needed to create the dataset
        features = [f if isinstance(f, str) else f.__dict__ for f in features] if features else None
        r = make_request(self._FS_URL, Endpoints.TRAINING_SET_FROM_VIEW, RequestType.POST, self._auth,
                         params={"view": training_view, "pks": return_pk_cols, "ts": return_ts_col,
                                 'save_as': save_as, 'return_type': ReturnType.map_to_request(return_type)},
                         body={"features": features, "start_time": start_time, "end_time": end_time},
                         headers={"Accept-Encoding": "gzip"})
        sql = r["sql"]
        tvw = TrainingView(**r["training_view"])
        features = [Feature(**f) for f in r["features"]]
        create_time = r['metadata']['training_set_create_ts']
        start_time = r['metadata']['training_set_start_ts']
        end_time = r['metadata']['training_set_end_ts']
        # Link this to mlflow for reproducibility and model deployment
        if self.mlflow_ctx and not return_sql:
            # These will only exist if the user called "save_as" otherwise they will be None
            training_set_id = r['metadata'].get('training_set_id')
            training_set_version = r['metadata'].get('training_set_version')
            self.link_training_set_to_mlflow(features, create_time, start_time, end_time, tvw,
                                             training_set_id=training_set_id,
                                             training_set_version=training_set_version, training_set_name=save_as)

        return _format_training_set_output(response=r, return_type=return_type, splice_ctx=self.splice_ctx)

    def list_training_sets(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary a training sets available, with the map name -> description. If there is no description,
        the value will be an emtpy string

        :return: Dict[str, Optional[str]]
        """
        raise NotImplementedError("To see available training views, run fs.describe_training_views()")

    def create_feature_set(self, schema_name: str, table_name: str, primary_keys: Dict[str, str],
                           desc: Optional[str] = None, features: Optional[List[Feature]] = None) -> FeatureSet:
        """
        Creates and returns a new feature set

        :param schema_name: The schema under which to create the feature set table
        :param table_name: The table name for this feature set
        :param primary_keys: The primary key column(s) of this feature set
        :param desc: The (optional) description
        :param features: An optional list of features. If provided, the Features will be created with the Feature Set
        :Example:
            .. code-block:: python

                from splicemachine.features import FeatureType, Feature
                f1 = Feature(
                    name='my_first_feature',
                    description='the first feature',
                    feature_data_type='INT',
                    feature_type=FeatureType.ordinal,
                    tags=['good_feature','a new tag', 'ordinal'],
                    attributes={'quality':'awesome'}
                )
                f2 = Feature(
                    name='my_second_feature',
                    description='the second feature',
                    feature_data_type='FLOAT',
                    feature_type=FeatureType.continuous,
                    tags=['not_as_good_feature','a new tag'],
                    attributes={'quality':'not as awesome'}
                )
                feats = [f1, f2]
                feature_set = fs.create_feature_set(
                    schema_name='splice',
                    table_name='foo',
                    primary_keys={'MOMENT_KEY':"INT"},
                    desc='test fset',
                    features=feats
                )

        :return: FeatureSet
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        features = [f.__dict__ for f in features] if features else None
        fset_dict = {"schema_name": schema_name,
                     "table_name": table_name,
                     "primary_keys": {pk: sql_to_datatype(primary_keys[pk]) for pk in primary_keys},
                     "description": desc,
                     "features": features}

        print(f'Registering feature set {schema_name}.{table_name} in Feature Store')
        if features:
            print(f'Registering {len(features)} features for {schema_name}.{table_name} in the Feature Store')
        r = make_request(self._FS_URL, Endpoints.FEATURE_SETS, RequestType.POST, self._auth, body=fset_dict)
        return FeatureSet(**r)

    def update_feature_set(self, schema_name: str, table_name: str, primary_keys: Dict[str, str],
                           desc: Optional[str] = None, features: Optional[List[Feature]] = None) -> FeatureSet:
        """
        Creates and returns a new version of an existing feature set. Use this method when you want to make changes
        to a deployed feature set.

        :param schema_name: The schema under which to create the feature set table
        :param table_name: The table name for this feature set
        :param primary_keys: The primary key column(s) of this feature set
        :param desc: The (optional) description
        :param features: An optional list of features. If provided, any non-existant Features will be created with the Feature Set
        :Example:
            .. code-block:: python

                from splicemachine.features import FeatureType, Feature
                f1 = Feature(
                    name='my_first_feature',
                    description='the first feature',
                    feature_data_type='INT',
                    feature_type=FeatureType.ordinal,
                    tags=['good_feature','a new tag', 'ordinal'],
                    attributes={'quality':'awesome'}
                )
                f2 = Feature(
                    name='my_second_feature',
                    description='the second feature',
                    feature_data_type='FLOAT',
                    feature_type=FeatureType.continuous,
                    tags=['not_as_good_feature','a new tag'],
                    attributes={'quality':'not as awesome'}
                )
                feats = [f1, f2]
                feature_set = fs.update_feature_set(
                    schema_name='splice',
                    table_name='foo',
                    primary_keys={'MOMENT_KEY':"INT"},
                    desc='test fset',
                    features=feats
                )

        :return: FeatureSet
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        features = [f.__dict__ for f in features] if features else None
        fset_dict = {"primary_keys": {pk: sql_to_datatype(primary_keys[pk]) for pk in primary_keys},
                     "description": desc,
                     "features": features}

        print(f'Registering feature set {schema_name}.{table_name} in Feature Store')
        if features:
            print(f'Registering {len(features)} features for {schema_name}.{table_name} in the Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.FEATURE_SETS}/{schema_name}.{table_name}', RequestType.PUT,
                         self._auth, body=fset_dict)
        return FeatureSet(**r)

    def alter_feature_set(self, schema_name: str, table_name: str, primary_keys: Optional[Dict[str, str]] = None,
                          desc: Optional[str] = None, version: Optional[Union[str, int]] = None) -> FeatureSet:
        """
        Alters the specified (or default latest) version of a feature set, if that version is not yet deployed. Use this method when you want to make changes to
        an undeployed version of a feature set, or when you want to change version-independant metadata, such as description.

        :param schema_name: The schema under which to create the feature set table
        :param table_name: The table name for this feature set
        :param primary_keys: The primary key column(s) of this feature set
        :param desc: The (optional) description
        :param version: The version you wish to alter (number or 'latest'). If None, will default to the latest undeployed version
        :return: FeatureSet
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        params = {'version': version}
        fset_dict = {
            "primary_keys": {pk: sql_to_datatype(primary_keys[pk]) for pk in primary_keys} if primary_keys else None,
            "description": desc}

        print(f'Registering feature set {schema_name}.{table_name} in Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.FEATURE_SETS}/{schema_name}.{table_name}', RequestType.PATCH,
                         self._auth,
                         params=params, body=fset_dict)
        return FeatureSet(**r)

    def update_feature_metadata(self, name: str, desc: Optional[str] = None, tags: Optional[List[str]] = None,
                                attributes: Optional[Dict[str, str]] = None):
        """
        Update the metadata of a feature

        :param name: The feature name
        :param desc: The (optional) feature description (default None)
        :param tags: (optional) List of (str) tag words (default None)
        :param attributes: (optional) Dict of (str) attribute key/value pairs (default None)
        :return: updated Feature
        """
        f_dict = {"description": desc, 'tags': tags, "attributes": attributes}
        print(f'Registering feature {name} in Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.FEATURES}/{name}', RequestType.PUT, self._auth,
                         body=f_dict)
        f = Feature(**r)
        return f

    def create_feature(self, schema_name: str, table_name: str, name: str, feature_data_type: str,
                       feature_type: str, desc: str = None, tags: List[str] = None, attributes: Dict[str, str] = None):
        """
        Add a feature to a feature set

        :param schema_name: The feature set schema
        :param table_name: The feature set table name to add the feature to
        :param name: The feature name
        :param feature_data_type: The datatype of the feature. Must be a valid SQL datatype
        :param feature_type: splicemachine.features.FeatureType of the feature. The available types are from the FeatureType class: FeatureType.[categorical, ordinal, continuous].
            You can see available feature types by running

            .. code-block:: python

                    from splicemachine.features import FeatureType
                    print(FeatureType.get_valid())

        :param desc: The (optional) feature description (default None)
        :param tags: (optional) List of (str) tag words (default None)
        :param attributes: (optional) Dict of (str) attribute key/value pairs (default None)
        :return: Feature created
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        assert feature_type in FeatureType.get_valid(), f"The feature_type {feature_type} in not valid. Valid feature " \
                                                        f"types include {FeatureType.get_valid()}. Use the FeatureType" \
                                                        f" class provided by splicemachine.features"

        f_dict = {"name": name, "description": desc or '', "feature_data_type": sql_to_datatype(feature_data_type),
                  "feature_type": feature_type, "tags": tags, "attributes": attributes}
        print(f'Registering feature {name} in Feature Store')
        r = make_request(self._FS_URL, Endpoints.FEATURES, RequestType.POST, self._auth,
                         {"schema": schema_name, "table": table_name}, f_dict)
        f = Feature(**r)
        return f
        # TODO: Backfill the feature

    def create_training_view(self, name: str, sql: str, primary_keys: List[str], join_keys: List[str],
                             ts_col: str, label_col: Optional[str] = None, desc: Optional[str] = None) -> None:
        """
        Registers a training view for use in generating training SQL

        :param name: The training set name. This must be unique to other existing training sets
        :param sql: (str) a SELECT statement that includes:\n
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * join key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param replace: (Optional[bool]) Whether to replace an existing training view
        :param join_keys: (List[str]) A list of join keys in the sql that are used to get the desired features in
            get_training_set
        :param desc: (Optional[str]) An optional description of the training set
        :param verbose: Whether or not to print the SQL before execution (default False)
        :return:
        """
        assert name != "None", "Name of training view cannot be None!"

        tv_dict = {"name": name, "description": desc, "pk_columns": primary_keys, "ts_column": ts_col,
                   "label_column": label_col,
                   "join_columns": join_keys, "sql_text": sql}
        print(f'Registering Training View {name} in the Feature Store')
        make_request(self._FS_URL, Endpoints.TRAINING_VIEWS, RequestType.POST, self._auth, body=tv_dict)

    def update_training_view(self, name: str, sql: str, primary_keys: List[str], join_keys: List[str],
                             ts_col: str, label_col: Optional[str] = None, desc: Optional[str] = None) -> TrainingView:
        """
        Creates and returns a new version of a training view for use in generating training SQL. Use this function when you want to
        make changes to a training view without affecting its dependencies

        :param name: The training set name.
        :param sql: (str) a SELECT statement that includes:\n
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * join key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param replace: (Optional[bool]) Whether to replace an existing training view
        :param join_keys: (List[str]) A list of join keys in the sql that are used to get the desired features in
            get_training_set
        :param desc: (Optional[str]) An optional description of the training set
        :param verbose: Whether or not to print the SQL before execution (default False)
        :return:
        """
        assert name != "None", "Name of training view cannot be None!"

        tv_dict = {"description": desc, "pk_columns": primary_keys, "ts_column": ts_col, "label_column": label_col,
                   "join_columns": join_keys, "sql_text": sql}
        print(f'Registering Training View {name} in the Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.TRAINING_VIEWS}/{name}', RequestType.PUT, self._auth, body=tv_dict)
        return TrainingView(**r)

    def alter_training_view(self, name: str, sql: Optional[str] = None, primary_keys: Optional[List[str]] = None,
                            join_keys: Optional[List[str]] = None, ts_col: Optional[str] = None,
                            label_col: Optional[str] = None,
                            desc: Optional[str] = None, version: Optional[Union[str, int]] = None) -> TrainingView:
        """
        Alters an existing version of a training view. Use this method when you want to make changes to a version of a training view
        that has no dependencies, or when you want to change version-independent metadata, such as description.

        :param name: The training set name. This must be unique to other existing training sets unless replace is True
        :param sql: (str) a SELECT statement that includes:\n
            * the primary key column(s) - uniquely identifying a training row/case
            * the inference timestamp column - timestamp column with which to join features (temporal join timestamp)
            * join key(s) - the references to the other feature tables' primary keys (ie customer_id, location_id)
            * (optionally) the label expression - defining what the training set is trying to predict
        :param primary_keys: (List[str]) The list of columns from the training SQL that identify the training row
        :param ts_col: The timestamp column of the training SQL that identifies the inference timestamp
        :param label_col: (Optional[str]) The optional label column from the training SQL.
        :param join_keys: (List[str]) A list of join keys in the sql that are used to get the desired features in
            get_training_set
        :param desc: (Optional[str]) An optional description of the training set
        :param version: The version you wish to alter (number or 'latest'). If None, will default to the latest version
        :return:
        """
        assert name != "None", "Name of training view cannot be None!"

        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        params = {'version': version}
        tv_dict = {"description": desc, "pk_columns": primary_keys, "ts_column": ts_col, "label_column": label_col,
                   "join_columns": join_keys, "sql_text": sql}
        print(f'Registering Training View {name} in the Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.TRAINING_VIEWS}/{name}', RequestType.PATCH, self._auth,
                         params=params, body=tv_dict)
        return TrainingView(**r)

    def _process_features(self, features: List[Union[Feature, str]]) -> List[Feature]:
        """
        Process a list of Features parameter. If the list is strings, it converts them to Features, else returns itself

        :param features: The list of Feature names or Feature objects
        :return: List[Feature]
        """
        feat_str = [f for f in features if isinstance(f, str)]
        str_to_feat = self.get_features_by_name(names=feat_str, as_list=True) if feat_str else []
        all_features = str_to_feat + [f for f in features if not isinstance(f, str)]
        assert all(
            [isinstance(i, Feature) for i in all_features]), "It seems you've passed in Features that are neither" \
                                                             " a feature name (string) or a Feature object"
        return all_features

    def deploy_feature_set(self, schema_name: str, table_name: str, version: Union[str, int] = 'latest',
                           migrate: bool = False):
        """
        Deploys a feature set to the database. This persists the feature sets existence.
        As of now, once deployed you cannot delete the feature set or add/delete features.
        The feature set must have already been created with :py:meth:`~features.FeatureStore.create_feature_set`

        :param schema_name: The schema of the created feature set
        :param table_name: The table of the created feature set
        :param version: The version of the feature set to deploy
        :param migrate: Whether or not to migrate data from a past version of this feature set
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()
        print(f'Deploying Feature Set {schema_name}.{table_name}...', end=' ')
        make_request(self._FS_URL, Endpoints.DEPLOY_FEATURE_SET, RequestType.POST, self._auth,
                     {"schema": schema_name, "table": table_name, "version": version, "migrate": migrate})
        print('Done.')

    def get_features_from_feature_set(self, schema_name: str, table_name: str) -> List[Feature]:
        """
        Returns either a pandas DF of feature details or a List of features for a specified feature set.
        You can get features from multiple feature sets by concatenating the results of this call.
        For example, to get features from 2 feature sets, `foo.bar1` and `foo2.bar4`:

        .. code-block:: python

                features = fs.get_features_from_feature_set('foo','bar1') + fs.get_features_from_feature_set('foo2','bar4')

        If you want a list of just the Feature NAMES (ie a List[str]) you can simply run:

        .. code-block:: python

                features = fs.get_features_from_feature_set('foo','bar1') + fs.get_features_from_feature_set('foo2','bar4')
                feature_names = [f.name for f in features]

        :param schema_name: Feature Set schema name
        :param table_name: Feature Set table name
        :return: List of Features
        """
        r = make_request(self._FS_URL, Endpoints.FEATURE_SET_DETAILS, RequestType.GET, self._auth,
                         params={'schema': schema_name, 'table': table_name})
        features = [Feature(**feature) for feature in r.pop("features")]
        return features

    def describe_feature_sets(self) -> None:
        """
        Prints out a description of a all feature sets, with all features in the feature sets and whether the feature
        set is deployed

        :return: None
        """
        r = make_request(self._FS_URL, Endpoints.FEATURE_SET_DETAILS, RequestType.GET, self._auth)

        print('Available feature sets')
        for desc in r if type(r) == list else [r]:
            features = [Feature(**feature) for feature in desc.pop('features')]
            fset = FeatureSet(**desc)
            print('-' * 23)
            self._feature_set_describe(fset, features)

    def describe_feature_set(self, schema_name: str, table_name: str) -> None:
        """
        Prints out a description of a given feature set, with all features in the feature set and whether the feature
        set is deployed

        :param schema_name: feature set schema name
        :param table_name: feature set table name
        :return: None
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        r = make_request(self._FS_URL, Endpoints.FEATURE_SET_DETAILS, RequestType.GET, self._auth,
                         params={'schema': schema_name, 'table': table_name})
        desc = r
        if not desc: raise SpliceMachineException(
            f"Feature Set {schema_name}.{table_name} not found. Check name and try again.")
        features = [Feature(**feature) for feature in desc.pop("features")]
        fset = FeatureSet(**desc)
        self._feature_set_describe(fset, features)

    def _feature_set_describe(self, fset: FeatureSet, features: List[Feature]):
        print(f'{fset.schema_name}.{fset.table_name} - {fset.description}')
        print('Primary keys:', fset.primary_keys)
        print('\nAvailable features:')
        display(pd.DataFrame(f.__dict__ for f in features))

    def describe_training_views(self) -> None:
        """
        Prints out a description of all training views, the ID, name, description and optional label

        :param training_view: The training view name
        :return: None
        """
        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_DETAILS, RequestType.GET, self._auth)

        print('Available training views')
        for desc in r if type(r) == list else [r]:
            features = [Feature(**f) for f in desc.pop('features')]
            tcx = TrainingView(**desc)
            print('-' * 23)
            self._training_view_describe(tcx, features)

    def describe_training_view(self, training_view: str, version: Union[int, str] = 'latest') -> None:
        """
        Prints out a description of a given training view, the ID, name, description and optional label

        :param training_view: The training view name
        :param version: The training view version
        :return: None
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        r = make_request(self._FS_URL, Endpoints.TRAINING_VIEW_DETAILS, RequestType.GET, self._auth,
                         {'name': training_view, 'version': version})
        desc = r
        if not desc: raise SpliceMachineException(f"Training view {training_view} not found. Check name and try again.")

        feats = [Feature(**f) for f in desc.pop('features')]
        tcx = TrainingView(**desc)
        self._training_view_describe(tcx, feats)

    def _training_view_describe(self, tcx: TrainingView, feats: List[Feature]):
        print(f'ID({tcx.view_id}) {tcx.name} - {tcx.description} - LABEL: {tcx.label_column}')
        print(f'Available features in {tcx.name}:')

        col_order = ['name', 'description', 'feature_data_type', 'feature_set_name', 'feature_type', 'tags',
                     'last_update_ts',
                     'last_update_username', 'compliance_level', 'feature_set_id', 'feature_id']
        display(pd.DataFrame(f.__dict__ for f in feats)[col_order])

    def set_feature_description(self):
        raise NotImplementedError

    def get_training_set_from_deployment(self, schema_name: str, table_name: str, label: str = None,
                                         return_pk_cols: bool = False, return_ts_col: bool = False,
                                         return_type: str = 'spark'):
        """
        Reads Feature Store metadata to rebuild orginal training data set used for the given deployed model.

        :param schema_name: model schema name
        :param table_name: model table name
        :param label: An optional label to specify for the training set. If specified, the feature set of that feature
            will be used as the "anchor" feature set, meaning all point in time joins will be made to the timestamps of
            that feature set. This feature will also be recorded as a "label" feature for this particular training set
            (but not others in the future, unless this label is again specified).
        :param return_pk_cols: bool Whether or not the returned sql should include the primary key column(s)
        :param return_ts_cols: bool Whether or not the returned sql should include the timestamp column
        :param return_type: How the data should be returned. If not specified, a Spark DF will be returned.
            Available arguments are: 'spark', 'pandas', 'json', 'sql'
            sql will return the SQL necessary to generate the dataframe
        :return: SparkDF the Training Frame
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        if return_type not in ReturnType.get_valid():
            raise SpliceMachineException(f'Return type must be one of {ReturnType.get_valid()}')

        r = make_request(self._FS_URL, Endpoints.TRAINING_SET_FROM_DEPLOYMENT, RequestType.GET, self._auth,
                         {"schema": schema_name, "table": table_name, "label": label,
                          "pks": return_pk_cols, "ts": return_ts_col,
                          'return_type': ReturnType.map_to_request(return_type)})

        metadata = r['metadata']

        tv_name = metadata['name']
        start_time = metadata['training_set_start_ts']
        end_time = metadata['training_set_end_ts']
        create_time = metadata['training_set_create_ts']

        tv = TrainingView(**r['training_view']) if 'training_view' in r else None
        features = [Feature(**f) for f in r['features']]

        if self.mlflow_ctx:
            self.link_training_set_to_mlflow(features, create_time, start_time, end_time, tv)

        return _format_training_set_output(response=r, return_type=return_type, splice_ctx=self.splice_ctx)

    def remove_feature(self, name: str):
        """
        Removes a feature. This will run 2 checks.
            1. See if the feature exists.
            2. See if the feature belongs to a feature set that has already been deployed.

            If either of these are true, this function will throw an error explaining which check has failed

            :param name: feature name
            :return:
        """
        print(f"Removing feature {name}...", end=' ')
        make_request(self._FS_URL, f'{Endpoints.FEATURES}/{name}', RequestType.DELETE, self._auth)
        print('Done.')

    def get_deployments(self, schema_name: str = None, table_name: str = None, training_set: str = None,
                        feature: str = None, feature_set: str = None, version: Union[str, int] = None):
        """
        Returns a list of all (or specified) available deployments

        :param schema_name: model schema name
        :param table_name: model table name
        :param training_set: training set name
        :param feature: passing this in will return all deployments that used this feature
        :param feature_set: passing this in will return all deployments that used this feature set
        :param version: the version of the feature set parameter, if used
        :return: List[Deployment] the list of Deployments as dicts
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        return make_request(self._FS_URL, Endpoints.DEPLOYMENTS, RequestType.GET, self._auth,
                            {'schema': schema_name, 'table': table_name, 'name': training_set, 'feat': feature,
                             'fset': feature_set, 'version': version})

    def get_training_set_features(self, training_set: str = None):
        """
        Returns a list of all features from an available Training Set, as well as details about that Training Set

        :param training_set: training set name
        :return: TrainingSet as dict
        """
        r = make_request(self._FS_URL, Endpoints.TRAINING_SET_FEATURES, RequestType.GET, self._auth,
                         {'name': training_set})
        r['features'] = [Feature(**f) for f in r['features']]
        return r

    def remove_feature_set(self, schema_name: str, table_name: str, version: Union[str, int] = None,
                           purge: bool = False) -> None:
        """
        Deletes a feature set if appropriate. You can currently delete a feature set in two scenarios:
        1. The feature set has not been deployed
        2. The feature set has been deployed, but not linked to any training sets

        If both of these conditions are false, this will fail.

        Optionally set purge=True to force delete the feature set and all of the associated Training Sets using the
        Feature Set. ONLY USE IF YOU KNOW WHAT YOU ARE DOING. This will delete Training Sets, but will still fail if
        there is an active deployment with this feature set. That cannot be overwritten

        :param schema_name: The Feature Set Schema
        :param table_name: The Feature Set Table
        :param version: The Feature Set Version
        :param purge: Whether to force delete training sets that use the feature set (that are not used in deployments)
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        if purge:
            warnings.warn("You've set purge=True, I hope you know what you are doing! This will delete any dependent"
                          " Training Sets (except ones used in an active model deployment)")
        print(f'Removing Feature Set {schema_name}.{table_name}...', end=' ')
        make_request(self._FS_URL, Endpoints.FEATURE_SETS,
                     RequestType.DELETE, self._auth,
                     {"schema": schema_name, "table": table_name, "version": version, "purge": purge})
        print('Done.')

    def get_pipes(self, names: Optional[List[str]] = None) -> List[Pipe]:
        """
        Returns a list of pipes whose names are provided

        :param names: The list of pipe names
        :return: List[Pipe] The list of Pipe objects
        """
        r = make_request(self._FS_URL, Endpoints.PIPES, RequestType.GET, self._auth, {"name": names})
        return [Pipe(**p, splice_ctx=self.splice_ctx) for p in r]

    def get_pipe(self, name: str, version: Optional[Union[str, int]] = 'latest') -> Pipe:
        """
        Returns a pipe version

        :param name: The pipe name
        :param version: The version to return (either an int or 'latest'). Default is 'latest'
        :return: List[Pipe] The list of Pipe objects
        """
        assert version, "version cannot be none!"
        r = make_request(self._FS_URL, f'{Endpoints.PIPES}/{name}', RequestType.GET, self._auth, {"version": version})
        return Pipe(**r[0], splice_ctx=self.splice_ctx)

    def get_pipe_versions(self, name: str) -> List[Pipe]:
        """
        Returns a pipe version or list of pipe versions for a provided pipe name

        :param name: The pipe name
        :return: List[Pipe] The list of Pipe objects
        """
        r = make_request(self._FS_URL, f'{Endpoints.PIPES}/{name}', RequestType.GET, self._auth)
        return [Pipe(**p, splice_ctx=self.splice_ctx) for p in r]

    def create_pipe(self, name: str, ptype: str, lang: str, func: Callable, description: Optional[str] = None) -> Pipe:
        """
        Creates and returns a new pipe

        :param name: The pipe name. This must be unique to other pipes
        :param ptype: splicemachine.features.PipeType of the pipe. The available types are from the PipeType class: PipeType.[source, batch, online, realtime].
            You can see available feature types by running

            .. code-block:: python

                    from splicemachine.features import PipeType
                    print(PipeType.get_valid())

        :param lang: str The language of the pipe's function. Currently supported types are 'python', 'pyspark', and 'sql'
        :param func: The actual python function or sql query used in the transformation
        :param description: (Optional[str]) An optional description of the pipe
        :return:
        """
        assert ptype in PipeType.get_valid(), f"The pipe type {ptype} is not valid. Valid pipe " \
                                              f"types include {PipeType.get_valid()}. Use the PipeType" \
                                              f" class provided by splicemachine.features"

        assert lang in PipeLanguage.get_valid(), f"The pipe language {lang} is not supported. Currently supported" \
                                                 f"languages include {PipeLanguage.get_valid()}. Use the PipeLanguage" \
                                                 f" class provided by splicemachine.features"

        func.__globals__.pop('splice', None)
        func.__globals__.pop('spark', None)
        f = base64.encodebytes(cloudpickle.dumps(func)).decode('ascii').strip()
        p_dict = { "name": name, "description": description, "ptype": ptype, "lang": lang, "func": f, "code": getsource(func) }

        print(f'Registering Pipe {name} in the Feature Store')
        r = make_request(self._FS_URL, Endpoints.PIPES, RequestType.POST, self._auth, body=p_dict)
        return Pipe(**r, splice_ctx=self.splice_ctx)

    def update_pipe(self, name: str, func: Callable, description: Optional[str] = None) -> Pipe:
        """
        Creates and returns a new version of a pipe. Use this function when you want to
        make changes to a pipe without affecting its dependencies

        :param name: The pipe name.
        :param func: The actual python function or sql query used in the transformation
        :param description: (Optional[str]) An optional description of the pipe
        :return:
        """
        assert name != "None", "Name of pipe cannot be None!"

        func.__globals__.pop('splice', None)
        func.__globals__.pop('spark', None)
        f = base64.encodebytes(cloudpickle.dumps(func)).decode('ascii').strip()
        p_dict = { "description": description, "func": f, "code": getsource(func) }

        print(f'Updating Pipe {name} in the Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.PIPES}/{name}', RequestType.PUT, self._auth, body=p_dict)
        return Pipe(**r, splice_ctx=self.splice_ctx)

    def alter_pipe(self, name: str, func: Optional[Union[str, Callable]] = None, description: Optional[str] = None,
                   version: Union[int, str] = 'latest') -> Pipe:
        """
        Alters an existing version of a pipe. Use this method when you want to make changes to a version of a pipe
        that has no dependencies, or when you want to change version-independent metadata, such as description.

        :param name: The pipe name.
        :param func: The actual python function or sql query used in the transformation
        :param description: (Optional[str]) An optional description of the pipe
        :param version: The version to alter (either an int or 'latest'). Default 'latest'
        :return:
        """
        assert name != "None", "Name of pipe cannot be None!"
        assert func or description, "Please enter either a function or description"
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        if func:
            func.__globals__.pop('splice', None)
            func.__globals__.pop('spark', None)
            f = base64.encodebytes(cloudpickle.dumps(func)).decode('ascii').strip()
            c = getsource(func)
        else:
            f = None
            c = None

        p_dict = { "description": description, "func": f , "code": c}
        params = { "version" : version }

        print(f'Altering Pipe {name} in the Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.PIPES}/{name}', RequestType.PATCH, self._auth, params=params,
                         body=p_dict)
        return Pipe(**r, splice_ctx=self.splice_ctx)

    def remove_pipe(self, name: str, version: Union[int, str] = None) -> None:
        """
        Removes an existing version of a pipe. If no versions remain, deletes pipe metadata.

        :param name: The pipe name.
        :param version: The version to remove (either an int or 'latest'). If None, removes all versions
        :return:
        """
        assert name != "None", "Name of pipe cannot be None!"
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        p_dict = {"version": version}

        print(f'Removing Pipe {name} from the Feature Store')
        make_request(self._FS_URL, f'{Endpoints.PIPES}/{name}', RequestType.DELETE, self._auth, params=p_dict)

    def get_pipelines(self, names: Optional[List[str]] = None) -> List[Pipeline]:
        """
        Returns a list of pipelines whose names are provided

        :param names: The list of pipeline names
        :return: List[Pipline] The list of Pipeline objects
        """
        r = make_request(self._FS_URL, Endpoints.PIPELINES, RequestType.GET, self._auth, { "name": names })
        pipelines = []
        for pl in r:
            pl['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in pl['pipes']]
            pipelines.append(Pipeline(**pl))
        return pipelines

    def get_pipeline(self, name: str, version: Optional[Union[str, int]] = 'latest') -> Pipeline:
        """
        Returns a pipeline version

        :param name: The pipeline name
        :param version: The version to return (either an int or 'latest'). Default is 'latest'
        :return: List[Pipeline] The list of Pipeline objects
        """
        assert version, "version cannot be none!"
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")
        
        r = make_request(self._FS_URL, f'{Endpoints.PIPELINES}/{name}', RequestType.GET, self._auth, { "version": version })
        pl = r[0]
        pl['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in pl['pipes']]
        return Pipeline(**pl)

    def get_pipeline_versions(self, name: str) -> List[Pipeline]:
        """
        Returns a pipeline version or list of pipeline versions for a provided pipeline name

        :param name: The pipeline name
        :return: List[Pipeline] The list of Pipeline objects
        """
        r = make_request(self._FS_URL, f'{Endpoints.PIPELINES}/{name}', RequestType.GET, self._auth)
        pipelines = []
        for pl in r:
            pl['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in pl['pipes']]
            pipelines.append(Pipeline(**pl))
        return pipelines

    def _get_deployed_pipelines(self) -> List[Pipeline]:
        """
        Returns a list of deployed pipeline versions

        :param names: The list of pipeline names
        :return: List[Pipline] The list of Pipeline objects
        """
        r = make_request(self._FS_URL, Endpoints.PIPELINES, RequestType.GET, self._auth, { "deployed": True })
        pipelines = []
        for pl in r:
            pl['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in pl['pipes']]
            pipelines.append(Pipeline(**pl))
        return pipelines

    def create_pipeline(self, name: str, pipeline_start_date: datetime, pipes: List[Pipe], pipeline_interval: Optional[str] = None, description: Optional[str] = None) -> Pipeline:
        """
        Creates and returns a new pipeline

        :param name: The pipeline name. This must be unique to other pipelines
        :param pipeline_start_date: The start date of the pipe when deployed. This can be in the past, if a backfill is desired, or in the future. 
                                    Any granularity smaller than a day will be ignored.
        :param pipes: An (ordered) list of the pipes that make up this pipeline
        :param pipeline_interval: (Optional[str]) The interval at which to run the pipeline. Either a cron expression or an Airflow cron preset. 
                                   If None, the pipeline will be need to be triggered manually
        :param description: (Optional[str]) An optional description of the pipeline
        :return:
        """
        pipes = [p if isinstance(p, str) else p._to_json() for p in pipes]
        p_dict = { "name": name, "description": description, "pipeline_start_date": str(pipeline_start_date), "pipeline_interval": pipeline_interval, "pipes": pipes }

        print(f'Registering Pipeline {name} in the Feature Store')
        r = make_request(self._FS_URL, Endpoints.PIPELINES, RequestType.POST, self._auth, body=p_dict)
        r['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in r['pipes']]
        return Pipeline(**r)

    def update_pipeline(self, name: str, pipeline_start_date: datetime, pipes: List[Pipe], pipeline_interval: Optional[str] = None, description: Optional[str] = None) -> Pipeline:
        """
        Creates and returns a new version of a pipeline. Use this function when you want to
        make changes to a pipeline without affecting its dependencies

        :param name: The pipeline name. This must be unique to other pipelines
        :param pipeline_start_date: The start date of the pipe when deployed. This can be in the past, if a backfill is desired, or in the future. 
                                    Any granularity smaller than a day will be ignored.
        :param pipes: An (ordered) list of the pipes that make up this pipeline
        :param pipeline_interval: (Optional[str]) The interval at which to run the pipeline. Either a cron expression or an Airflow cron preset. 
                                   If None, the pipeline will be need to be triggered manually
        :param description: (Optional[str]) An optional description of the pipeline
        :return:
        """
        assert name != "None", "Name of pipeline cannot be None!"

        pipes = [p if isinstance(p, str) else p._to_json() for p in pipes]
        p_dict = { "description": description, "pipeline_start_date": str(pipeline_start_date), "pipeline_interval": pipeline_interval, "pipes": pipes }

        print(f'Updating Pipeline {name} in the Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.PIPELINES}/{name}', RequestType.PUT, self._auth, body=p_dict)
        r['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in r['pipes']]
        return Pipeline(**r)

    def alter_pipeline(self, name: str, pipeline_start_date: Optional[date] = None, pipes: Optional[List[Union[str, Pipe]]] = None, pipeline_interval: Optional[str] = None,
                    description: Optional[str] = None, version: Union[int, str] = 'latest') -> Pipe:
        """
        Alters an existing version of a pipeline. Use this method when you want to make changes to a version of a pipeline
        that has no dependencies, or when you want to change version-independent metadata, such as description.

        :param name: The pipeline name. This must be unique to other pipelines
        :param pipeline_start_date: The start date of the pipe when deployed. This can be in the past, if a backfill is desired, or in the future. 
                                    Any granularity smaller than a day will be ignored.
        :param pipes: An (ordered) list of the pipes that make up this pipeline
        :param pipeline_interval: (Optional[str]) The interval at which to run the pipeline. Either a cron expression or an Airflow cron preset. 
                                   If None, the pipeline will be need to be triggered manually
        :param description: (Optional[str]) An optional description of the pipeline
        :param version: The version to alter (either an int or 'latest'). Default 'latest'
        :return:
        """
        assert name != "None", "Name of pipeline cannot be None!"
        assert pipeline_start_date or pipeline_interval or pipes or description, "At least one attribute must be entered to alter"
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        pipes = [p if isinstance(p, str) else p._to_json() for p in pipes] if pipes else None
        p_dict = { "description": description, "pipeline_start_date": str(pipeline_start_date) if pipeline_start_date else None, "pipeline_interval": pipeline_interval, "pipes": pipes }
        params = { "version" : version }

        print(f'Altering Pipeline {name} in the Feature Store')
        r = make_request(self._FS_URL, f'{Endpoints.PIPELINES}/{name}', RequestType.PATCH, self._auth, params=params, body=p_dict)
        r['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in r['pipes']]
        return Pipeline(**r)

    def remove_pipeline(self, name: str, version: Union[int, str] = None) -> None:
        """
        Removes an existing version of a pipeline. If no versions remain, deletes pipeline metadata.

        :param name: The pipeline name.
        :param version: The version to remove (either an int or 'latest'). If None, removes all versions
        :return:
        """
        assert name != "None", "Name of pipeline cannot be None!"
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")

        p_dict = { "version": version }

        print(f'Removing Pipe {name} from the Feature Store')
        make_request(self._FS_URL, f'{Endpoints.PIPELINES}/{name}', RequestType.DELETE, self._auth, params=p_dict)

    def deploy_pipeline(self, name: str, schema_name: str, table_name: str, version: Union[str, int] = 'latest'):
        """
        Deploys a pipeline to a specified feature set. This schedules the pipeline as an Airflow DAG, with each pipe as its own task

        :param name: The name of the pipeline
        :param schema_name: The schema of the feature set
        :param table_name: The table of the feature set
        :param version: The version of the pipeline to deploy
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()
        print(f'Deploying Pipeline {name}...',end=' ')
        r = make_request(self._FS_URL, Endpoints.DEPLOY_PIPELINE, RequestType.POST, self._auth, { "name": name, "schema": schema_name, "table": table_name, "version": version })
        print('Done.')
        r['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in r['pipes']]
        return Pipeline(**r)

    def undeploy_pipeline(self, name: str, version: Union[str, int] = 'latest'):
        """
        Undeploys a pipeline to a specified feature set. This unschedules the associated Airflow DAG

        :param name: The name of the pipeline
        :param version: The version of the pipeline to undeploy
        """
        if isinstance(version, str) and version != 'latest':
            raise SpliceMachineException("Version parameter must be a number or 'latest'")
        print(f'Undeploying Pipeline {name}...',end=' ')
        r = make_request(self._FS_URL, Endpoints.UNDEPLOY_PIPELINE, RequestType.POST, self._auth, { "name": name, "version": version })
        print('Done.')
        r['pipes'] = [Pipe(**p, splice_ctx=self.splice_ctx) for p in r['pipes']]
        return Pipeline(**r)

    def _retrieve_model_data_sets(self, schema_name: str, table_name: str):
        """
        Returns the training set dataframe and model table dataframe for a given deployed model.

        :param schema_name: model schema name
        :param table_name: model table name
        :return:
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        training_set_df = self.get_training_set_from_deployment(schema_name, table_name)
        model_table_df = self.splice_ctx.df(f'SELECT * FROM {schema_name}.{table_name}')
        return training_set_df, model_table_df

    def _retrieve_training_set_metadata_from_deployement(self, schema_name: str, table_name: str):
        """
        Reads Feature Store metadata to retrieve definition of training set used to train the specified model.
        :param schema_name: model schema name
        :param table_name: model table name
        :return:
        """
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        sql = SQL.get_deployment_metadata.format(schema_name=schema_name, table_name=table_name)
        deploy_df = self.splice_ctx.df(sql).collect()
        cnt = len(deploy_df)
        if cnt == 1:
            return deploy_df[0]

    def display_model_feature_drift(self, schema_name: str, table_name: str):
        """
        Displays feature by feature comparison between the training set of the deployed model and the input feature
        values used with the model since deployment.

        :param schema_name: name of database schema where model table is deployed
        :param table_name: name of the model table
        :return: None
        """
        try:
            from .utils.drift_utils import build_feature_drift_plot
        except:
            raise SpliceMachineException('You must have matplotlib installed to call this function')

        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        metadata = make_request(self._FS_URL, Endpoints.TRAINING_SET_FROM_DEPLOYMENT, RequestType.GET,
                                self._auth, params={"schema": schema_name, "table": table_name})['metadata']

        training_set_df, model_table_df = self._retrieve_model_data_sets(schema_name, table_name)
        features = [f.upper() for f in metadata['features'].split(',')]
        build_feature_drift_plot(features, training_set_df, model_table_df)

    def display_model_drift(self, schema_name: str, table_name: str, time_intervals: int,
                            start_time: datetime = None, end_time: datetime = None):
        """
        Displays as many as `time_intervals` plots showing the distribution of the model prediction within each time
        period. Time periods are equal periods of time where predictions are present in the model table
        `schema_name.table_name`. Model predictions are first filtered to only those occurring after `start_time` if
        specified and before `end_time` if specified.

        :param schema_name: schema where the model table resides
        :param table_name: name of the model table
        :param time_intervals: number of time intervals to plot
        :param start_time: if specified, filters to only show predictions occurring after this date/time
        :param end_time: if specified, filters to only show predictions occurring before this date/time
        """
        try:
            from .utils.drift_utils import build_model_drift_plot
        except:
            raise SpliceMachineException('You must have matplotlib installed to call this function')
       
        # database stores object names in upper case
        schema_name = schema_name.upper()
        table_name = table_name.upper()

        # set default timeframe if not specified
        if not start_time:
            start_time = datetime(1900, 1, 1, 0, 0, 0)
        if not end_time:
            end_time = datetime.now()
        # retrieve predictions the model has made over time
        sql = SQL.get_model_predictions.format(schema_name=schema_name, table_name=table_name,
                                               start_time=start_time, end_time=end_time)
        model_table_df = self.splice_ctx.df(sql)
        build_model_drift_plot(model_table_df, time_intervals)

    def display_feature_search(self, pandas_profile=True):
        """
        Returns an interactive feature search that enables users to search for features and profiles the selected Feature.
        Two forms of this search exist. 1 for use inside of the managed Splice Machine notebook environment, and one
        for standard Jupyter. This is because the managed Splice Jupyter environment has extra functionality that would
        not be present outside of it. The search will be automatically rendered depending on the environment.

        :param pandas_profile: Whether to use pandas / spark to profile the feature. If pandas is selected
        but the dataset is too large, it will fall back to Spark. Default Pandas.
        """
        # Need IPython for this function
        try:
            from splicemachine.notebook import _in_splice_compatible_env
        except:
            raise SpliceMachineException('You must be running in IPython or Jupyter with IPython installed to use this function')
        # It may have the attr but be None
        if not (hasattr(self, 'splice_ctx') and isinstance(self.splice_ctx, PySpliceContext)):
            raise SpliceMachineException('You must register a Splice Machine Context (PySpliceContext) in order to use '
                                         'this function currently')
        # This will show a warning if packages are missing, and we don't want that at the top level to always be shown
        from splicemachine.features.utils.search_utils import feature_search_external, feature_search_internal
        if _in_splice_compatible_env():
            feature_search_internal(self, pandas_profile)
        else:
            feature_search_external(self, pandas_profile)

    def __get_pipeline(self, df, features, label, model_type):
        """
        Creates a Pipeline with preprocessing steps (StringINdexer, VectorAssembler) for each feature depending
        on feature type, and returns the pipeline for training for feature elimination

        :param df: Spark Dataframe
        :param features: List[Feature] to train on
        :param label: Label name to train on
        :param model_type: (str) the model type - avl options are "classification" and "regression"
        :return: Unfit Spark Pipeline
        """
        categorical_features = [f.name for f in features if f.is_categorical()]
        numeric_features = [f.name for f in features if f.is_continuous() or f.is_ordinal()]
        indexed_features = [f'{n}_index' for n in categorical_features]

        si = [StringIndexer(inputCol=n, outputCol=f'{n}_index', handleInvalid='keep') for n in categorical_features]
        all_features = numeric_features + indexed_features

        v = VectorAssembler(inputCols=all_features, outputCol='features', handleInvalid='keep')
        if model_type == 'classification':
            si += [StringIndexer(inputCol=label, outputCol=f'{label}_index', handleInvalid='keep')]
            clf = RandomForestClassifier(labelCol=f'{label}_index')
        else:
            clf = RandomForestRegressor(labelCol=label)
        return MLPipeline(stages=si + [v, clf]).fit(df)

    def __get_feature_importance(self, feature_importances, df, features_column):
        """
        Gets the ordered feature importance for the feature elimination rounds
        :param feature_importances: Spark model featureImportances attribute
        :param df: Spark dataframe
        :param features_column: Column name of the dataframe that holds the features
        :return: Sorted pandas dataframe with the feature importances and feature names
        """
        feature_rank = []
        for i in df.schema[features_column].metadata["ml_attr"]["attrs"]:
            feature_rank += df.schema[features_column].metadata["ml_attr"]["attrs"][i]
        features_df = pd.DataFrame(feature_rank)
        features_df['score'] = features_df['idx'].apply(lambda x: feature_importances[x])
        return (features_df.sort_values('score', ascending=False))

    def __log_mlflow_results(self, name, rounds, mlflow_results):
        """
        Logs the results of feature elimination to mlflow

        :param name: MLflow run name
        :param rounds: Number of rounds of feature elimination that were run
        :param mlflow_results: The params / metrics to log
        :return:
        """
        try:
            if self.mlflow_ctx.active_run():
                self.mlflow_ctx.start_run(run_name=name)
            for r in range(rounds):
                with self.mlflow_ctx.start_run(run_name=f'Round {r}', nested=True):
                    self.mlflow_ctx.log_metrics(mlflow_results[r])
        finally:
            self.mlflow_ctx.end_run()

    def __prune_features_for_elimination(self, features) -> List[Feature]:
        """
        Removes incompatible features from the provided list if they are not compatible with SparkML modeling

        :param features: List[Feature] the provided list
        :return: List[Features] the pruned list
        """
        from splicemachine.spark.constants import SQL_MODELING_TYPES
        invalid_features = {f for f in features if f.feature_data_type['data_type'] not in SQL_MODELING_TYPES}
        valid_features = list(set(features) - invalid_features)
        if invalid_features: print('The following features are invalid for modeling based on their Data Types:\n')
        for f in invalid_features:
            print(f.name, f.feature_data_type)
        return valid_features

    def run_feature_elimination(self, df, features: List[Union[str, Feature]], label: str = 'label', n: int = 10,
                                verbose: int = 0, model_type: str = 'classification', step: int = 1,
                                log_mlflow: bool = False, mlflow_run_name: str = None,
                                return_importances: bool = False):

        """
        Runs feature elimination using a Spark decision tree on the dataframe passed in. Optionally logs results to mlflow

        :param df: The dataframe with features and label
        :param features: The list of feature names (or Feature objects) to run elimination on
        :param label: the label column names
        :param n: The number of features desired. Default 10
        :param verbose: The level of verbosity. 0 indicated no printing. 1 indicates printing remaining features after
            each round. 2 indicates print features and relative importances after each round. Default 0
        :param model_type: Whether the model to test with will be a regression or classification model. Default classification
        :param log_mlflow: Whether or not to log results to mlflow as nested runs. Default false
        :param mlflow_run_name: The name of the parent run under which all subsequent runs will live. The children run
            names will be {mlflow_run_name}_{num_features}_features. ie testrun_5_features, testrun_4_features etc
        :return:
        """

        train_df = df
        features = self._process_features(features)
        remaining_features = self.__prune_features_for_elimination(features)
        rnd = 0
        mlflow_results = []
        assert len(
            remaining_features) > n, \
            "You've set the number of desired features (n) greater than the number of available features"
        while len(remaining_features) > n:
            rnd += 1
            num_features = max(len(remaining_features) - step, n)  # Don't go less than the specified value
            print(f'Building {model_type} model')
            model = self.__get_pipeline(train_df, remaining_features, label, model_type)
            print('Getting feature importance')
            feature_importances = self.__get_feature_importance(model.stages[-1].featureImportances,
                                                                model.transform(train_df), "features").head(
                num_features)
            remaining_features_and_label = list(feature_importances['name'].values) + [label]
            train_df = train_df.select(*remaining_features_and_label)
            remaining_features = [f for f in remaining_features if f.name in feature_importances['name'].values]
            print(f'{len(remaining_features)} features remaining')

            if verbose == 1:
                print(f'Round {rnd} complete. Remaining Features:')
                for i, f in enumerate(list(feature_importances['name'].values)):
                    print(f'{i}. {f}')
            elif verbose == 2:
                print(f'Round {rnd} complete. Remaining Features:')
                display(feature_importances.reset_index(drop=True))

            # Add results to a list for mlflow logging
            round_metrics = {'Round': rnd, 'Number of features': len(remaining_features)}
            for index, row in feature_importances.iterrows():
                round_metrics[row['name']] = row['score']
            mlflow_results.append(round_metrics)

        if log_mlflow and self.mlflow_ctx:
            run_name = mlflow_run_name or f'feature_elimination_{label}'
            self.__log_mlflow_results(run_name, rnd, mlflow_results)

        return remaining_features, feature_importances.reset_index(
            drop=True) if return_importances else remaining_features

    def link_training_set_to_mlflow(self, features: Union[List[Feature], List[str]], create_time: datetime,
                                    start_time: datetime = None,
                                    end_time: datetime = None, tvw: TrainingView = None,
                                    current_values_only: bool = False,
                                    training_set_id: Optional[int] = None, training_set_version: Optional[int] = None,
                                    training_set_name: Optional[str] = None):

        # Here we create a null training view and pass it into the training set. We do this because this special kind
        # of training set isn't standard. It's not based on a training view, on primary key columns, a label column,
        # or a timestamp column . This is simply a joined set of features from different feature sets.
        # But we still want to track this in mlflow as a user may build and deploy a model based on this. So we pass in
        # a null training view that can be tracked with a "name" (although the name is None). This is a likely case
        # for (non time based) clustering use cases.

        if not tvw:
            tvw = TrainingView(pk_columns=[], ts_column=None, label_column=None, view_sql=None, name=None,
                               description=None)
        ts = TrainingSet(training_view=tvw, features=features, create_time=create_time,
                         start_time=start_time, end_time=end_time, training_set_id=training_set_id,
                         training_set_version=training_set_version, training_set_name=training_set_name)

        # If the user isn't getting historical values, that means there isn't really a start_time, as the user simply
        # wants the most up to date values of each feature. So we set start_time to end_time (which is datetime.today)
        if current_values_only:
            ts.start_time = ts.end_time

        self.mlflow_ctx._active_training_set: TrainingSet = ts
        ts._register_metadata(self.mlflow_ctx)

    def set_feature_store_url(self, url: str):
        """
        Sets the Feature Store URL. You must call this before calling any feature store functions, or set the FS_URL
        environment variable before creating your Feature Store object

        :param url: The Feature Store URL
        """
        self._FS_URL = url

    def login_fs(self, username, password):
        """
        Function to login to the Feature Store using basic auth. These correspond to your Splice Machine database user
        and password. If you are running outside of the managed Splice Machine Cloud Service, you must call either
        this or set_token in order to call any functions in the feature store, or by setting the SPLICE_JUPYTER_USER and
        SPLICE_JUPYTER_PASSWORD environments variable before creating your FeatureStore object.

        :param username: Username
        :param password: Password
        """
        self._auth = HTTPBasicAuth(username, password)

    def set_token(self, token):
        """
        Function to login to the Feature Store using JWT. This corresponds to your Splice Machine database user's JWT
        token. If you are running outside of the managed Splice Machine Cloud Service, you must call either
        this or login_fs in order to call any functions in the feature store, or by setting the SPLICE_JUPYTER_TOKEN
        environment variable before creating your FeatureStore object.

        :param token: JWT Token
        """
        self._auth = token

    def __try_auto_login(self):
        """
        Tries to login the user  automatically. This will only work if the user is not
        using the cloud service.

        :return: None
        """
        token = _get_token()
        if token:
            self.set_token(token)
            return

        user, password = _get_credentials()
        if user and password:
            self.login_fs(user, password)
