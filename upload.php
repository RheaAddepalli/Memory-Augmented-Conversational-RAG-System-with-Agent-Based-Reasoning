<?php
header("Content-Type: application/json; charset=utf-8");

if ($_SERVER["REQUEST_METHOD"] !== "POST") {
    echo json_encode([
        "status" => "error",
        "message" => "Invalid request."
    ]);
    exit;
}

if (!isset($_FILES["file"])) {
    echo json_encode([
        "status" => "error",
        "message" => "No file uploaded."
    ]);
    exit;
}

$saveChoice = $_POST["saveChoice"] ?? "yes";
$targetDir  = __DIR__ . "/uploads/";
$fileName   = basename($_FILES["file"]["name"]);
$targetFile = $targetDir . $fileName;
$fileType   = strtolower(pathinfo($targetFile, PATHINFO_EXTENSION));

/* -----------------------
   VALIDATIONS
------------------------ */
if ($fileType !== "pdf") {
    echo json_encode([
        "status" => "error",
        "message" => "Only PDF files are allowed."
    ]);
    exit;
}

if ($_FILES["file"]["error"] !== UPLOAD_ERR_OK) {
    echo json_encode([
        "status" => "error",
        "message" => "Upload error: " . $_FILES["file"]["error"]
    ]);
    exit;
}

/* -----------------------
   ENSURE UPLOAD FOLDER
------------------------ */
if (!is_dir($targetDir)) {
    mkdir($targetDir, 0777, true);
}

/* -----------------------
   MOVE FILE
------------------------ */
if (!move_uploaded_file($_FILES["file"]["tmp_name"], $targetFile)) {
    echo json_encode([
        "status" => "error",
        "message" => "Failed to move uploaded file."
    ]);
    exit;
}

/* -----------------------
   RUN PYTHON
------------------------ */
$pythonExe = escapeshellarg(__DIR__ . "/venv_gai_old/Scripts/python.exe");
$script    = escapeshellarg(__DIR__ . "/process_pdf.py");
$pdfArg    = escapeshellarg($targetFile);

$command = "$pythonExe $script $pdfArg";
$output  = shell_exec($command . " 2>&1");

/* -----------------------
   DEBUG LOG
------------------------ */
file_put_contents(
    __DIR__ . "/debug.txt",
    "COMMAND:\n$command\n\nOUTPUT:\n$output\n\n",
    FILE_APPEND
);

/* -----------------------
   PARSE PYTHON JSON
------------------------ */
$output = trim($output, "\xEF\xBB\xBF \n\r\t");

if (preg_match('/\{.*\}/s', $output, $m)) {
    $decoded = json_decode($m[0], true);
} else {
    echo json_encode([
        "status" => "error",
        "message" => "Invalid response from Python.",
        "raw" => $output
    ]);
    exit;
}

if (!isset($decoded["summary"]) || trim($decoded["summary"]) === "") {
    echo json_encode([
        "status" => "error",
        "message" => "Summary is empty."
    ]);
    exit;
}

/* -----------------------
   DELETE IF USER SAID NO
------------------------ */
$pdfHash   = md5_file($targetFile);
$cacheFile = __DIR__ . "/saved_summaries/$pdfHash.json";

if ($saveChoice === "no") {
    if (file_exists($targetFile)) {
        unlink($targetFile);
    }
    if (file_exists($cacheFile)) {
        unlink($cacheFile);
    }
}

/* -----------------------
   SUCCESS RESPONSE
------------------------ */
echo json_encode([
    "status"  => "success",
    "summary" => $decoded["summary"],
    "cached"  => $decoded["cached"] ?? false,
    "saved"   => $saveChoice
]);
