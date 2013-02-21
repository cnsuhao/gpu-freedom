<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="DOCUMENT">
        <html>
            <head>
			    <meta http-equiv="refresh" content="60" />
                <title>GPU Server - List Clients (refreshs each minute)</title>
            </head>
            <body>
			    <img src="../images/gpu-inverse.jpg" border="0" />
                <h2>List online nodes</h2>
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