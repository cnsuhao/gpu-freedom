<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:include href="head.inc.xsl"/>
    <xsl:include href="menu.inc.xsl"/>
    <xsl:include href="bottom.inc.xsl"/>
    
    <xsl:template match="jobdefinitions">
        <html>
            <head>
			    <title>GPU Server - List Job Definitions</title>
				<link rel="stylesheet" type="text/css" href="../stylesheet/gpu.css" />
            </head>
            <body>
                <table>
                <tr>
                <xsl:call-template name="HEAD"/>
                </tr>
                <tr>
                <xsl:call-template name="MENU"/>
                <td valign="top">
			   
                <h2>List Job Definitions</h2>
                <table border="1">
					<tr>
						<th>id</th>
						<th>jobdefinitionid</th>
						<th>job</th>
						<th>jobtype</th>
						<th>nodename</th>
						<th>create_dt</th>
						<th>update_dt</th>
					</tr>
                    <!-- msg loop -->
                    <xsl:apply-templates select="jobdefinition"/>
                </table>
                
                <xsl:call-template name="BOTTOM"/>
                </td>
                </tr>
                </table>
				
				
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="jobdefinition">
        <tr bgcolor="#81F781">
            <td>
                <xsl:value-of select="id"/>			
            </td>
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
                <xsl:value-of select="nodename"/>			
            </td>
			<td>
                <xsl:value-of select="create_dt"/>			
            </td>
			<td>
                <xsl:value-of select="update_dt"/>			
            </td>
		</tr>
    </xsl:template>
 
</xsl:stylesheet>