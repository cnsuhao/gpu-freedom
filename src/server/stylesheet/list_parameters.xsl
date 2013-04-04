<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="parameters">
        <html>
            <head>
                <title>GPU Server - Parameters</title>
            </head>
            <body>
			    <a href="../index.php"><img src="../images/gpu-inverse.jpg" border="0"></img></a>
                <h2>Parameters</h2>
                <table border="1">
					<tr>
						<th>id</th>
						<th>type</th>
						<th>name</th>
						<th>value</th>
						<th>create_dt</th>
						<th>update_dt</th>
					</tr>
                    
                    <xsl:apply-templates select="parameter"/>
                </table>
				<hr />
				<a href="../index.php">Back</a><br />
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="parameter">
        <tr>
            <td>
                <xsl:value-of select="id"/>			
            </td>
			<td>
                <xsl:value-of select="paramtype"/>			
            </td>
			<td>
                <xsl:value-of select="paramname"/>			
            </td>
			<td>
                <xsl:value-of select="paramvalue"/>					
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