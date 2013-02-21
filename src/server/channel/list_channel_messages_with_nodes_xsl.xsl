<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="channel">
        <html>
            <head>
			    <meta http-equiv="refresh" content="20" />
                <title>GPU Server - List latest channel messages (refreshes each 20 seconds)</title>
            </head>
            <body>
			    <img src="../images/gpu-inverse.jpg" border="0" />
                <h2>List latest channel messages</h2>
                <table border="1">
					<tr>
						<th>id</th>
						<th>channel type</th>
						<th>channel name</th>
						<th>nodename</th>
						<th>user</th>
						<th>content</th>
						<th>create datum</th>
						<th>country</th>
						<th>longitude</th>
						<th>latitude</th>
					</tr>
                    <!-- msg loop -->
                    <xsl:apply-templates select="msg"/>
                </table>
            </body>
        </html>
    </xsl:template>
 
    <!-- msg template -->
    <xsl:template match="msg">
        <!-- display msg field -->
        <tr>
            <td>
                <xsl:value-of select="id"/>			
            </td>
			<td>
                <xsl:value-of select="chantype"/>			
            </td>
			<td>
                <xsl:value-of select="channame"/>			
            </td>
			<td>
                <xsl:value-of select="nodename"/>			
            </td>
			<td>
                <xsl:value-of select="user"/>			
            </td>
			<td>
                <xsl:value-of select="content"/>			
            </td>
			<td>
                <xsl:value-of select="create_dt"/>			
            </td>	
			
			<!-- client loop -->
			<xsl:apply-templates select="client"/>
		</tr>
    </xsl:template>
 
    <!-- client template -->
    <xsl:template match="client">
            <td><xsl:value-of select="country"/></td>
            <td><xsl:value-of select="longitude"/></td>
			<td><xsl:value-of select="latitude"/></td>
    </xsl:template>
 
</xsl:stylesheet>