import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import re
from datetime import datetime

class DataRefinery:
    def __init__(self, raw_data):
        self.df = raw_data.copy()
        self.logs = []

    def _add_log(self, stage, message):
        now = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{now}] {stage}: {message}")

    def _smart_date_processing(self):
        date_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    date_cols.append(col)
                except: continue
        
        for col in date_cols:
            self.df[f'{col}_day'] = self.df[col].dt.day
            self.df[f'{col}_month'] = self.df[col].dt.month
            self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
            self.df[f'{col}_is_weekend'] = self.df[col].dt.dayofweek.isin([5, 6]).astype(int)
            self.df.drop(columns=[col], inplace=True)
            self._add_log("Temporal", f"Synthesized time-features from {col}")

    def _privacy_guard(self):
        mask_count = 0
        for col in self.df.columns:
            col_l = col.lower()
            if any(k in col_l for k in ['mail', 'phone', 'tel', 'tc', 'address']):
                self.df[col] = self.df[col].apply(lambda x: "HIDDEN_" + str(hash(str(x)))[:6])
                mask_count += 1
        if mask_count: self._add_log("Privacy", f"Anonymized {mask_count} sensitive columns")

    def _feature_engineering(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].nunique() > 20:
                self.df[f'{col}_sq'] = self.df[col] ** 2
                self._add_log("Synthesis", f"Generated non-linear feature for {col}")

    def _clean_and_impute(self):
        self.df.drop(columns=[c for c in self.df.columns if self.df[c].nunique() <= 1], inplace=True)
        
        for col in self.df.select_dtypes(exclude=[np.number]).columns:
            self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))

        if self.df.isnull().sum().sum() > 0:
            self.df[self.df.columns] = IterativeImputer(max_iter=5, random_state=42).fit_transform(self.df)
            self._add_log("Refinement", "Applied ML-based data restoration")

    def process(self):
        self._add_log("System", "Golden Standard Pipeline Activated")
        
        self._privacy_guard()
        self._smart_date_processing()
        self._clean_and_impute()
        self._feature_engineering()
        
        target_guess = self.df.columns[-1]
        X = self.df.drop(columns=[target_guess])
        y = self.df[target_guess]
        
        scaler = PowerTransformer(method='yeo-johnson')
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        selector = SelectFromModel(RandomForestClassifier(n_estimators=30, random_state=42))
        selector.fit(X_scaled, y.astype(int))
        X_final = X_scaled[X_scaled.columns[selector.get_support()]]
        
        self.df = pd.concat([X_final, y.reset_index(drop=True)], axis=1)
        self._add_log("Success", f"Output optimized for Golden ML Standards. Target: {target_guess}")
        return {"refined_data": self.df, "logs": self.logs}