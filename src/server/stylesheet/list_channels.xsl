<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="channeltypes">
        <html>
            <head>
                <title>GPU Server - Available channels</title>
            </head>
            <body>
			    <a href="../index.php"><img src="../images/gpu-inverse.jpg" border="0" /></a>
                <h2>Available Channels</h2>
                <table border="1">
					<tr>
						<th>channel name</th>
						<th>channel type</th>
					</tr>
                    
                    <xsl:apply-templates select="channeltype"/>
                </table>
				<hr />
				<a href="../index.php">Back</a><br />
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="channeltype">
        <tr>
            <td>
                <xsl:value-of select="channame"/>			
            </td>
			<td>
                <xsl:value-of select="chantype"/>			
            </td>
		</tr>
    </xsl:template>
 
</xsl:stylesheet>