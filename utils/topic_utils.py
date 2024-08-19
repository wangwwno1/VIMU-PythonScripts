from typing import Sequence, Optional, Mapping, Union, List

import pandas as pd


class TopicInfo:
    def __init__(
            self,
            topic_name,
            fill_method: str = 'backward',
            max_instace: int = 1,
            timestamp_column: str = 'timestamp',
            data_columns: Sequence[str] = tuple(),
            rename_map: Optional[Mapping] = None
    ):
        self.fill_method: Optional[str] = fill_method
        self.max_instance = max(max_instace, 1)
        self.timestamp_column = timestamp_column
        self.data_columns = data_columns
        self.rename_map = rename_map

        self.topic_name: str = topic_name

    @property
    def renamed_columns(self) -> Union[None, Sequence[str]]:
        if self.data_columns is None:
            return None

        result = []
        for k in self.data_columns:
            result.append(self.rename_map[k] if k in self.rename_map else k)
        return result

    @property
    def topic_keys(self):
        for inst in range(self.max_instance):
            yield f"{self.topic_name}_{inst}"

    def extract_dataframes(self, udict: Mapping[str, pd.DataFrame]) -> List[pd.DataFrame]:
        result = []

        for name in self.topic_keys:
            if name not in udict:
                continue
            df = udict[name].copy()
            if self.timestamp_column != 'timestamp':
                df.set_index(pd.to_timedelta(df[self.timestamp_column], unit='us'), inplace=True, drop=True)
            # df.index.rename(f"{self.timestamp_column}_{name}", inplace=True)
            if len(self.data_columns) > 0:
                df = df.loc[:, self.data_columns]

            if self.rename_map is not None:
                df = df.rename(columns=self.rename_map)

            result.append(df)

        return result
