Fatal error: Call to undefined function curl_init()

you're using xampp as for your error shows it, locate php.ini in xampp directory probably located in C:\Program Files\xampp\php\php.ini and search for ;extension=php_curl.dll remove the ; to uncomment it. Restart xampp.