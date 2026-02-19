<?php
header("Content-Type: text/plain");

$cmd = "\"C:\\xampp\\htdocs\\GenAI-Doc-old\\venv_gai_old\\Scripts\\python.exe\" -c \"print('HELLO FROM APACHE')\" 2>&1";

echo shell_exec($cmd);
