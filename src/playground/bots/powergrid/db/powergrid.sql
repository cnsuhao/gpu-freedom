-- phpMyAdmin SQL Dump
-- version 3.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jul 02, 2013 at 10:16 AM
-- Server version: 5.5.25a
-- PHP Version: 5.4.4

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `powergrid`
--

-- --------------------------------------------------------

--
-- Table structure for table `tbfrequency`
--

CREATE TABLE IF NOT EXISTS `tbfrequency` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `frequencyhz` double DEFAULT NULL,
  `networkdiff` double DEFAULT NULL,
  `controlarea` varchar(16) COLLATE latin1_general_ci NOT NULL,
  `tso` varchar(16) COLLATE latin1_general_ci NOT NULL,
  `create_dt` datetime NOT NULL,
  `create_user` varchar(16) COLLATE latin1_general_ci DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=88 ;

--
-- Dumping data for table `tbfrequency`
--

INSERT INTO `tbfrequency` (`id`, `frequencyhz`, `networkdiff`, `controlarea`, `tso`, `create_dt`, `create_user`) VALUES
(86, 50, 26.788, 'SG_ST', 'Swissgrid', '2013-07-02 10:14:52', 'script'),
(87, 50.009, 26.655, 'SG_ST', 'Swissgrid', '2013-07-02 10:16:11', 'script');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
