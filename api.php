<?php

header("Content-Type: application/json; charset=UTF-8");

$outputFile = __DIR__ . DIRECTORY_SEPARATOR .
              "outputs" . DIRECTORY_SEPARATOR .
              "v7_final_ranking.csv";


if (!file_exists($outputFile)) {

    http_response_code(500);

    echo json_encode([
        "success" => false,
        "error" => "V7 recommendation file was not found.",
        "path_checked" => $outputFile
    ]);

    exit;
}


$handle = fopen($outputFile, "r");

if ($handle === false) {

    http_response_code(500);

    echo json_encode([
        "success" => false,
        "error" => "Unable to read V7 recommendation file."
    ]);

    exit;
}


$headers = fgetcsv($handle);

if ($headers === false) {

    fclose($handle);

    http_response_code(500);

    echo json_encode([
        "success" => false,
        "error" => "V7 recommendation file is empty."
    ]);

    exit;
}


$rows = [];


while (($data = fgetcsv($handle)) !== false) {

    if (count($data) !== count($headers)) {
        continue;
    }

    $row = array_combine($headers, $data);

    if ($row === false) {
        continue;
    }

    $rows[] = $row;
}


fclose($handle);


/*
 * V7 already performs the candidate selection
 * and ranking.
 *
 * We only expose the ML-supported recommendations
 * to the frontend at this stage.
 */

$recommendations = [];


foreach ($rows as $row) {

    if (
        isset($row["candidate_tier"]) &&
        trim($row["candidate_tier"]) === "A_ML_SUPPORTED"
    ) {

        $recommendations[] = $row;
    }
}


/*
 * Keep the V7 ordering.
 */

usort(
    $recommendations,
    function ($a, $b) {

        $rankA = isset($a["recommendation_rank"])
            ? (int)$a["recommendation_rank"]
            : PHP_INT_MAX;

        $rankB = isset($b["recommendation_rank"])
            ? (int)$b["recommendation_rank"]
            : PHP_INT_MAX;

        return $rankA <=> $rankB;
    }
);


/*
 * Frontend currently displays top 6.
 */

$recommendations = array_slice(
    $recommendations,
    0,
    6
);


echo json_encode(
    [
        "success" => true,
        "count" => count($recommendations),
        "recommendations" => $recommendations
    ],
    JSON_UNESCAPED_UNICODE |
    JSON_PRETTY_PRINT
);

?>