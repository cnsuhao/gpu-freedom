<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
                
    <xsl:include href="head.inc.xsl"/>
    <xsl:include href="menu.inc.xsl"/>
    <xsl:include href="bottom.inc.xsl"/>
 
    <xsl:template match="jobstats">
        <html>
            <head>
                <title>GPU Server - Job Statistics</title>
            </head>
            <body>
                <table>
                <tr>
                <xsl:call-template name="HEAD"/>
                </tr>
                <tr>
                <xsl:call-template name="MENU"/>
                <td valign="top">
			    
                <h2>Job Statistics</h2>
                <table border="1">
					<tr>
						<th>jobid</th>
						<th>job</th>
						<th>jobtype</th>
						<th>requiresack</th>
						<th>create_dt</th>
						<th>requests</th>
						<th>transmitted</th>
						<th>acknowledged</th>
						<th>received</th>
					</tr>
                    
                    <xsl:apply-templates select="jobstat"/>
                </table>
                
                <xsl:call-template name="BOTTOM"/>
                </td>
                </tr>
                </table>
				
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="jobstat">
        <tr bgcolor="#81F781">
            <td>
                <xsl:value-of select="jobdefinitionid"/>			
            </td>
			<td>
                <xsl:value-of select="job"/>			
            </td>
			<td>
                <xsl:value-of select="jobtype"/>			
            </td>
			<td>
                <xsl:value-of select="requireack"/>			
            </td>
			<td>
                <xsl:value-of select="create_dt"/>			
            </td>
			<td>
                <xsl:value-of select="requests"/>			
            </td>
			<td>
                <xsl:value-of select="transmitted"/>			
            </td>
			<td>
                <xsl:value-of select="acknowledged"/>			
            </td>
			<td>
                <xsl:value-of select="received"/>			
            </td>
		</tr>
    </xsl:template>
 
</xsl:stylesheet>