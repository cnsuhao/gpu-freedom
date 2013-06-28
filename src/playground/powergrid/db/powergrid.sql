-- phpMyAdmin SQL Dump
-- version 3.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jun 28, 2013 at 03:25 PM
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
-- Table structure for table `frequency`
--

CREATE TABLE IF NOT EXISTS `frequency` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `frequency` double NOT NULL,
  `networkdiff` double DEFAULT NULL,
  `controlarea` varchar(16) NOT NULL,
  `tso` varchar(16) NOT NULL,
  `create_dt` datetime NOT NULL,
  `create_user` varchar(16) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=5 ;

--
-- Dumping data for table `frequency`
--

INSERT INTO `frequency` (`id`, `frequency`, `networkdiff`, `controlarea`, `tso`, `create_dt`, `create_user`) VALUES
(2, 50.007, 13.201, 'SG_ST', 'Swissgrid', '2013-06-28 15:23:06', 'php'),
(3, 50.013, 13.14, 'SG_ST', 'Swissgrid', '2013-06-28 15:23:56', 'php'),
(4, 50.01, 13.149, 'SG_ST', 'Swissgrid', '2013-06-28 15:24:22', 'php');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
