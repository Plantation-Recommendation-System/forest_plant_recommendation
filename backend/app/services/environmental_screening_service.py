import numpy as np
import pandas as pd


# =========================================================
# ORDERING
# =========================================================

CLIMATE_TIER_ORDER = {
    "A_strict_match": 0,
    "B_near_match": 1,
    "C_partial_match": 2,
    "D_weak_match": 3,
}

SOIL_MATCH_ORDER = {
    "strong_texture_match": 0,
    "moderate_texture_match": 1,
    "weak_texture_match": 2,
    "insufficient_soil_evidence": 3,
}


class EnvironmentalScreeningService:

    # -----------------------------------------------------
    # APPLY HIERARCHICAL CLIMATE + SOIL ORDER
    # -----------------------------------------------------

    def rank_environmentally(
        self,
        matched_candidates: pd.DataFrame,
    ) -> pd.DataFrame:

        required = [
            "project_species_resolved",
            "climate_rank",
            "climate_tier",
            "soil_texture_match_pct",
            "soil_texture_match_label",
        ]

        missing = [
            col
            for col in required
            if col not in matched_candidates.columns
        ]

        if missing:
            raise ValueError(
                "Missing environmental columns: "
                f"{missing}"
            )

        df = matched_candidates.copy()

        # -------------------------------------------------
        # Soil questionnaire consistency
        #
        # Backwards-compatible:
        # if no questionnaire was supplied, do not reject
        # candidates just because the column is absent.
        # -------------------------------------------------

        if (
            "soil_questionnaire_consistency"
            not in df.columns
        ):
            df[
                "soil_questionnaire_consistency"
            ] = "questionnaire_unknown"

        # -----------------------------------------------
        # Internal sorting helpers
        # -----------------------------------------------

        df["_climate_order"] = (
            df["climate_tier"]
            .map(CLIMATE_TIER_ORDER)
            .fillna(99)
        )

        df["_soil_order"] = (
            df["soil_texture_match_label"]
            .map(SOIL_MATCH_ORDER)
            .fillna(99)
        )

        # -----------------------------------------------
        # HIERARCHICAL SORT
        #
        # 1. Climate tier
        # 2. Soil class
        # 3. Soil overlap %
        # 4. Original climate rank
        #
        # Questionnaire is a consistency gate,
        # not a weighted score.
        # -----------------------------------------------

        df = (
            df
            .sort_values(
                by=[
                    "_climate_order",
                    "_soil_order",
                    "soil_texture_match_pct",
                    "climate_rank",
                ],
                ascending=[
                    True,
                    True,
                    False,
                    True,
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        df["environmental_rank"] = np.arange(
            1,
            len(df) + 1,
        )

        # -----------------------------------------------
        # CATBOOST HANDOFF RULE
        #
        # Climate remains primary.
        #
        # A -> pass
        # B -> pass
        #
        # C -> requires:
        #      strong texture compatibility
        #      AND no direct disagreement between
        #      soil image and wet-soil observation
        #
        # D -> fallback only
        # -----------------------------------------------

        def handoff_status(row):

            climate = row[
                "climate_tier"
            ]

            soil = row[
                "soil_texture_match_label"
            ]

            consistency = str(
                row.get(
                    "soil_questionnaire_consistency",
                    "questionnaire_unknown",
                )
            ).strip().lower()

            if climate == "A_strict_match":
                return True

            if climate == "B_near_match":
                return True

            if (
                climate == "C_partial_match"
                and soil
                == "strong_texture_match"
                and consistency
                != "inconsistent"
            ):
                return True

            return False


        df["catboost_handoff"] = (
            df.apply(
                handoff_status,
                axis=1,
            )
        )

        # -----------------------------------------------
        # HUMAN-READABLE REASON
        # -----------------------------------------------

        def handoff_reason(row):

            climate = row[
                "climate_tier"
            ]

            soil = row[
                "soil_texture_match_label"
            ]

            consistency = str(
                row.get(
                    "soil_questionnaire_consistency",
                    "questionnaire_unknown",
                )
            ).strip().lower()

            if climate == "A_strict_match":

                if (
                    consistency
                    == "inconsistent"
                ):
                    return (
                        "Passed because climate match is "
                        "strict. Soil image and field "
                        "observation disagree, so soil "
                        "evidence should be interpreted "
                        "with caution."
                    )

                return (
                    "Passed because climate match is "
                    "strict; soil is secondary "
                    "supporting evidence."
                )

            if climate == "B_near_match":

                if (
                    consistency
                    == "inconsistent"
                ):
                    return (
                        "Passed because climate match is "
                        "near. Soil image and field "
                        "observation disagree, so soil "
                        "evidence is uncertain."
                    )

                return (
                    "Passed because climate match is "
                    "near; soil is secondary "
                    "supporting evidence."
                )

            if (
                climate
                == "C_partial_match"
                and soil
                == "strong_texture_match"
                and consistency
                != "inconsistent"
            ):

                return (
                    "Passed because partial climate "
                    "match is supported by strong soil "
                    "compatibility without contradictory "
                    "field texture evidence."
                )

            if (
                climate
                == "C_partial_match"
                and soil
                == "strong_texture_match"
                and consistency
                == "inconsistent"
            ):

                return (
                    "Not passed because the partial "
                    "climate match depended on strong "
                    "soil support, but the soil image "
                    "and wet-soil observation disagree."
                )

            if climate == "C_partial_match":

                return (
                    "Not passed: partial climate match "
                    "requires strong soil compatibility."
                )

            return (
                "Fallback only because climate match "
                "is weak."
            )


        df[
            "catboost_handoff_reason"
        ] = df.apply(
            handoff_reason,
            axis=1,
        )

        # Remove internal sort helpers
        df = df.drop(
            columns=[
                "_climate_order",
                "_soil_order",
            ]
        )

        return df


    # -----------------------------------------------------
    # ONLY SPECIES THAT GO INTO CATBOOST
    # -----------------------------------------------------

    def get_catboost_candidates(
        self,
        matched_candidates: pd.DataFrame,
    ) -> pd.DataFrame:

        ranked = (
            self.rank_environmentally(
                matched_candidates
            )
        )

        passed = (
            ranked[
                ranked[
                    "catboost_handoff"
                ]
            ]
            .copy()
            .sort_values(
                "environmental_rank"
            )
            .reset_index(drop=True)
        )

        return passed


environmental_screening_service = (
    EnvironmentalScreeningService()
)
