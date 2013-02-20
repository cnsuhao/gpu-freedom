<?php

// get current buffer contents and delete current output buffer
$xml_data = ob_get_clean();	

// define xsl file name from the script itself
$_name = explode('.', basename($_SERVER['SCRIPT_NAME']));
$xsl_file = current($_name) . '.xsl';

// create XSLT processor
$xp = new XsltProcessor();
// load the xml document and the xsl template
$xml = new DomDocument;
$xsl = new DomDocument;
$xml->loadXML($xml_data);
$xsl->load($xsl_file);

// load the xsl template
$xp->importStyleSheet($xsl);

// do XSL transformation and print result
print($xp->transformToXML($xml));

?>