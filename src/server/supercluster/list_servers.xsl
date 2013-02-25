<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="servers">
        <html>
            <head>
			    <meta http-equiv="refresh" content="180" />
                <title>GPU Server - List Servers (refreshs each three minutes)</title>
            </head>
            <body>
			    <img src="../images/gpu-inverse.jpg" border="0" />
                <h2>List Online Servers</h2>
                <table border="1">
					<tr>
						<th>id</th>
						<th>server name</th>
						<th>server url</th>
						<th>chat channel</th>
						<th>version</th>
						<th>longitude</th>
						<th>latitude</th>
						<th>uptime</th>
						<th>active nodes</th>
						<th>jobs in queue</th>
					</tr>
                    <!-- msg loop -->
                    <xsl:apply-templates select="server"/>
                </table>
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="server">
        <tr bgcolor="#F4FA58">
            <td>
                <xsl:value-of select="id"/>			
            </td>
			<td>
                <xsl:value-of select="servername"/>			
            </td>
			<td>
                <xsl:value-of select="serverurl"/>				
            </td>
			<td>
                <xsl:value-of select="chatchannel"/>			
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
                <xsl:value-of select="activenodes"/>			
            </td>
			<td>
                <xsl:value-of select="jobinqueue"/>			
            </td>
		</tr>
    </xsl:template>
 
</xsl:stylesheet>