<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="DOCUMENT">
        <html>
            <head>
                <title>GPU Server - List latest channel messages (refreshes each minute)</title>
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
            <td colspan="2">
                <xsl:value-of select="id"/>			
            </td>
			<td colspan="2">
                <xsl:value-of select="chantype"/>			
            </td>
			<td colspan="2">
                <xsl:value-of select="channame"/>			
            </td>
			<td colspan="2">
                <xsl:value-of select="nodename"/>			
            </td>
			<td colspan="2">
                <xsl:value-of select="user"/>			
            </td>
			<td colspan="2">
                <xsl:value-of select="content"/>			
            </td>
			<td colspan="2">
                <xsl:value-of select="create_dt"/>			
            </td>
			
        </tr>
        <!-- client loop -->
        <xsl:apply-templates select="client"/>
    </xsl:template>
 
    <!-- song template -->
    <xsl:template match="client">
        <tr>
            <td><xsl:value-of select="country"/></td>
            <td><xsl:value-of select="longitude"/></td>
			<td><xsl:value-of select="latitude"/></td>
        </tr>
    </xsl:template>
 
</xsl:stylesheet>