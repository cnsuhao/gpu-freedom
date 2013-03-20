<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="jobresults">
        <html>
            <head>
			    <title>GPU Server - List Job Results</title>
            </head>
            <body>
			    <a href="../index.php"><img src="../images/gpu-inverse.jpg" border="0" /></a>
                <h2>List Job Results</h2>
                <table border="1">
					<tr>
						<th>id</th>
						<th>jobresultid</th>
						<th>jobqueueid</th>
						<th>jobdefinitionid</th>
						<th>job</th>
						<th>job result</th>
						<th>workunit result</th>
						<th>wall time</th>
						<th>is erroneous</th>
						<th>errorid</th>
						<th>error argument</th>
						<th>error message</th>
						<th>answering nodename</th>
						<th>create_dt of request</th>
						<th>transmission_dt</th>
						<th>create_dt of result</th>
						<th>requester</th>
					</tr>
                    <xsl:apply-templates select="jobresult"/>
                </table>
				<hr />
				<a href="../index.php">Back</a><br />
            </body>
        </html>
    </xsl:template>

    <xsl:template match="jobresult">
        <tr bgcolor="#81F781">
            <td>
                <xsl:value-of select="id"/>			
            </td>
			<td>
                <xsl:value-of select="jobresultid"/>			
            </td>
			<td>
                <xsl:value-of select="jobqueueid"/>			
            </td>
			<td>
                <xsl:value-of select="jobdefinitionid"/>			
            </td>
			<td>
                <xsl:value-of select="job"/>			
            </td>
			<td>
                <xsl:value-of select="jobresult"/>			
            </td>
			<td>
                <xsl:value-of select="workunitresult"/>			
            </td>
			<td>
                <xsl:value-of select="walltime"/>			
            </td>
			<td>
                <xsl:value-of select="iserroneous"/>			
            </td>
			<td>
                <xsl:value-of select="errorid"/>			
            </td>	
			<td>
                <xsl:value-of select="errorarg"/>			
            </td>		
			<td>
                <xsl:value-of select="errormsg"/>			
            </td>		
			<td>
                <xsl:value-of select="nodename"/>			
            </td>		
			
			<xsl:apply-templates select="jobqueue"/>
		</tr>
    </xsl:template>
	
    <xsl:template match="jobqueue">
 			<td>
                <xsl:value-of select="create_dt"/>			
            </td>
			<td>
                <xsl:value-of select="transmission_dt"/>			
            </td>	
			<td>
                <xsl:value-of select="reception_dt"/>			
            </td>		
			<xsl:apply-templates select="jobdefinition"/>
   </xsl:template>
 
    <xsl:template match="jobdefinition">
            <td><xsl:value-of select="nodename"/></td>
    </xsl:template>
 
</xsl:stylesheet>