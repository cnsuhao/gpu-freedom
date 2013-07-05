<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
                
    <xsl:include href="head.inc.xsl"/>
    <xsl:include href="menu.inc.xsl"/>
    <xsl:include href="bottom.inc.xsl"/>
 
    <xsl:template match="stats">
        <html>
            <head>
                <title>GPU Server - Jobqueue Status</title>
            </head>
            <body>
			    
                <h2>Jobqueue Status</h2>
                <table border="1">
					<tr>
						<th>jobqueueid</th>
						<th>status</th>
						<th>message</th>
						<th>jobresultid</th>
						<th>nodename</th>
						<th>timestamp</th>
					</tr>
                    
                    <xsl:apply-templates select="jobstatus"/>
                </table>
				
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="jobstatus">
        <tr>
            <td>
                <xsl:value-of select="jobqueueid"/>			
            </td>
			<td>
                <xsl:value-of select="status"/>			
            </td>
			<td>
                <xsl:value-of select="message"/>			
            </td>
			<td>
                <xsl:value-of select="jobresultid"/>			
            </td>
			<td>
                <xsl:value-of select="nodename"/>			
            </td>
			<td>
                <xsl:value-of select="timestamp"/>					
            </td>
		</tr>
    </xsl:template>
 
</xsl:stylesheet>