<?php

function prepare_XSLT() {
	ob_start();
}

/**
 * apply_XSLT takes an XML and the XSL transforming into an HTML page
 *
 * Adapted from http://www.redips.net/php/from-mysql-to-html-with-xml/
 * Original version by Darko Bunic http://www.redips.net/about/
 * @param string  $basedir       - base folder of server, eg. "client", "cluster", "supercluster", etc.
 *
 */

function apply_XSLT($basedir) {
	
	// get current buffer contents and delete current output buffer
	$xml_data = ob_get_clean();	

	// define xsl file name from the script itself
	$_name = explode('.', basename($_SERVER['SCRIPT_NAME']));
	$xsl_file = current($_name) . '.xsl';

	// two useful debug statements:
	// echo "XSL file is: ../$basedir/$xsl_file";
    // echo "$xml_data";
	
	// create XSLT processor
	$xp = new XsltProcessor();
	// load the xml document and the xsl template
	$xml = new DomDocument;
	$xsl = new DomDocument;
	$xml->loadXML($xml_data);
	$xsl->load("../$basedir/$xsl_file");

	// load the xsl template
	$xp->importStyleSheet($xsl);

	// do XSL transformation and print result
	print($xp->transformToXML($xml));
	
	return;
}



?>