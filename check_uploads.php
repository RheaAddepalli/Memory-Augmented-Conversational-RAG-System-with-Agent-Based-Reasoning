<?php
$targetDir = "uploads/";

if (!is_dir($targetDir)) {
    echo "Error: 'uploads/' folder does NOT exist.<br>";
} elseif (!is_writable($targetDir)) {
    echo "Error: 'uploads/' folder is NOT writable.<br>";
} else {
    echo "'uploads/' folder is present and writable!<br>";
}
?>
