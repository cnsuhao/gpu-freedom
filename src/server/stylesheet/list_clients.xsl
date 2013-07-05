<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
                
    <xsl:include href="head.inc.xsl"/>
    <xsl:include href="menu.inc.xsl"/>
    <xsl:include href="bottom.inc.xsl"/>
 
    <xsl:template match="clients">
        <html>
            <head>
			    <meta http-equiv="refresh" content="60" />
                <title>GPU Server - List Clients (refreshs each minute)</title>
            </head>
            <body>
                <table>
                <tr>
                <xsl:call-template name="HEAD"/>
                </tr>
                <tr>
                <xsl:call-template name="MENU"/>
                <td valign="top">
			    
                <h2>List Online Clients</h2>
                <table border="1">
					<tr>
						<th>nodename</th>
						<th>country</th>
						<th>city</th>
						<th>os</th>
						<th>version</th>
						<th>longitude</th>
						<th>latitude</th>
						<th>uptime</th>
						<th>total uptime</th>
					</tr>
                    <!-- msg loop -->
                    <xsl:apply-templates select="client"/>
                </table>
				
                <xsl:call-template name="BOTTOM"/>
                </td>
                </tr>
                </table>
				
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="client">
        <tr bgcolor="#99CCFF">
            <td>
                <xsl:value-of select="nodename"/>			
            </td>
			<td>
                <xsl:value-of select="country"/>			
            </td>
			<td>
                <xsl:value-of select="city"/>			
            </td>
			<td>
                <xsl:value-of select="os"/>			
            </td>
			<td>
                <xsl:value-of select="version"/>			
            </td>
			<td>
                <xsl:value-of select="longitude"/>			
            </td>
			<td>
                <xsl:value-of select="latitude"/>			
            </td>
			<td>
                <xsl:value-of select="uptime"/>			
            </td>				
			
			<td>
                <xsl:value-of select="totaluptime"/>			
            </td>	
		</tr>
    </xsl:template>
 
</xsl:stylesheet>