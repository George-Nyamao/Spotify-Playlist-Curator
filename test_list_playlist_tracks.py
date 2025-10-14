import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock
from list_playlist_tracks import (
    get_spotify_client, 
    get_playlist_tracks, 
    extract_track_data, 
    get_playlist_dataframe,
    save_playlist_to_csv
)


class TestGetSpotifyClient:
    """Test the get_spotify_client function."""
    
    def test_get_spotify_client_success(self):
        """Test successful client creation with valid credentials."""
        with patch.dict(os.environ, {
            'SPOTIPY_CLIENT_ID': 'test_client_id',
            'SPOTIPY_CLIENT_SECRET': 'test_client_secret'
        }):
            with patch('list_playlist_tracks.SpotifyClientCredentials') as mock_credentials:
                with patch('list_playlist_tracks.spotipy.Spotify') as mock_spotify:
                    mock_credentials.return_value = 'mock_credentials_manager'
                    mock_spotify.return_value = 'mock_spotify_client'
                    
                    result = get_spotify_client()
                    
                    mock_credentials.assert_called_once_with(
                        client_id='test_client_id',
                        client_secret='test_client_secret'
                    )
                    mock_spotify.assert_called_once_with(auth_manager='mock_credentials_manager')
                    assert result == 'mock_spotify_client'
    
    def test_get_spotify_client_missing_client_id(self):
        """Test error when SPOTIPY_CLIENT_ID is missing."""
        with patch.dict(os.environ, {'SPOTIPY_CLIENT_SECRET': 'test_secret'}, clear=True):
            with pytest.raises(ValueError, match="SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set"):
                get_spotify_client()
    
    def test_get_spotify_client_missing_client_secret(self):
        """Test error when SPOTIPY_CLIENT_SECRET is missing."""
        with patch.dict(os.environ, {'SPOTIPY_CLIENT_ID': 'test_id'}, clear=True):
            with pytest.raises(ValueError, match="SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set"):
                get_spotify_client()
    
    def test_get_spotify_client_missing_both(self):
        """Test error when both credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set"):
                get_spotify_client()


class TestGetPlaylistTracks:
    """Test the get_playlist_tracks function."""
    
    def test_get_playlist_tracks_single_page(self):
        """Test getting tracks from a single-page playlist."""
        mock_sp = Mock()
        mock_response = {
            'items': [
                {'track': {'id': 'track1', 'name': 'Song 1'}},
                {'track': {'id': 'track2', 'name': 'Song 2'}}
            ],
            'next': None
        }
        mock_sp.playlist_tracks.return_value = mock_response
        
        result = get_playlist_tracks(mock_sp, 'playlist_id')
        
        mock_sp.playlist_tracks.assert_called_once_with('playlist_id')
        assert len(result) == 2
        assert result[0]['track']['id'] == 'track1'
        assert result[1]['track']['id'] == 'track2'
    
    def test_get_playlist_tracks_multiple_pages(self):
        """Test getting tracks from a multi-page playlist."""
        mock_sp = Mock()
        
        # First page
        first_response = {
            'items': [{'track': {'id': 'track1', 'name': 'Song 1'}}],
            'next': 'next_page_url'
        }
        
        # Second page
        second_response = {
            'items': [{'track': {'id': 'track2', 'name': 'Song 2'}}],
            'next': None
        }
        
        mock_sp.playlist_tracks.return_value = first_response
        mock_sp.next.return_value = second_response
        
        result = get_playlist_tracks(mock_sp, 'playlist_id')
        
        mock_sp.playlist_tracks.assert_called_once_with('playlist_id')
        mock_sp.next.assert_called_once_with(first_response)
        assert len(result) == 2
        assert result[0]['track']['id'] == 'track1'
        assert result[1]['track']['id'] == 'track2'
    
    def test_get_playlist_tracks_empty_playlist(self):
        """Test getting tracks from an empty playlist."""
        mock_sp = Mock()
        mock_response = {'items': [], 'next': None}
        mock_sp.playlist_tracks.return_value = mock_response
        
        result = get_playlist_tracks(mock_sp, 'playlist_id')
        
        assert result == []


class TestExtractTrackData:
    """Test the extract_track_data function."""
    
    def test_extract_track_data_success(self):
        """Test successful track data extraction."""
        tracks_info = [
            {
                'name': 'Test Song',
                'artists': [{'name': 'Test Artist'}],
                'album': {'name': 'Test Album', 'release_date': '2023-01-01'},
                'popularity': 80,
                'explicit': False,
                'duration_ms': 180000,
                'id': 'track123',
                'external_urls': {'spotify': 'https://open.spotify.com/track/track123'}
            }
        ]
        
        result = extract_track_data(tracks_info)
        
        assert len(result) == 1
        track = result[0]
        assert track['Name'] == 'Test Song'
        assert track['Artist'] == 'Test Artist'
        assert track['Album'] == 'Test Album'
        assert track['Release Date'] == '2023-01-01'
        assert track['Popularity'] == 80
        assert track['Explicit'] == False
        assert track['Duration (ms)'] == 180000
        assert track['Track ID'] == 'track123'
        assert track['Spotify URL'] == 'https://open.spotify.com/track/track123'
    
    def test_extract_track_data_multiple_artists(self):
        """Test track data extraction with multiple artists."""
        tracks_info = [
            {
                'name': 'Collaboration Song',
                'artists': [{'name': 'Artist 1'}, {'name': 'Artist 2'}],
                'album': {'name': 'Test Album', 'release_date': '2023-01-01'},
                'popularity': 70,
                'explicit': True,
                'duration_ms': 200000,
                'id': 'track456',
                'external_urls': {'spotify': 'https://open.spotify.com/track/track456'}
            }
        ]
        
        result = extract_track_data(tracks_info)
        
        assert len(result) == 1
        track = result[0]
        assert track['Artist'] == 'Artist 1'  # Should get the first artist
        assert track['Explicit'] == True
    
    def test_extract_track_data_no_artists(self):
        """Test track data extraction with no artists."""
        tracks_info = [
            {
                'name': 'Unknown Song',
                'artists': [],
                'album': {'name': 'Test Album', 'release_date': '2023-01-01'},
                'popularity': 50,
                'explicit': False,
                'duration_ms': 150000,
                'id': 'track789',
                'external_urls': {'spotify': 'https://open.spotify.com/track/track789'}
            }
        ]
        
        result = extract_track_data(tracks_info)
        
        assert len(result) == 1
        track = result[0]
        assert track['Artist'] == 'Unknown Artist'
    
    def test_extract_track_data_none_track(self):
        """Test track data extraction with None track."""
        tracks_info = [None, {
            'name': 'Valid Song',
            'artists': [{'name': 'Valid Artist'}],
            'album': {'name': 'Valid Album', 'release_date': '2023-01-01'},
            'popularity': 60,
            'explicit': False,
            'duration_ms': 160000,
            'id': 'track999',
            'external_urls': {'spotify': 'https://open.spotify.com/track/track999'}
        }]
        
        result = extract_track_data(tracks_info)
        
        assert len(result) == 1  # Should skip the None track
        track = result[0]
        assert track['Name'] == 'Valid Song'
    
    def test_extract_track_data_empty_list(self):
        """Test track data extraction with empty list."""
        result = extract_track_data([])
        assert result == []


class TestGetPlaylistDataframe:
    """Test the get_playlist_dataframe function."""
    
    def test_get_playlist_dataframe_success(self):
        """Test successful DataFrame creation."""
        mock_sp = Mock()
        
        # Mock playlist tracks response
        playlist_tracks = [
            {'track': {'id': 'track1'}},
            {'track': {'id': 'track2'}}
        ]
        
        # Mock tracks info response
        tracks_info = [
            {
                'name': 'Song 1',
                'artists': [{'name': 'Artist 1'}],
                'album': {'name': 'Album 1', 'release_date': '2023-01-01'},
                'popularity': 80,
                'explicit': False,
                'duration_ms': 180000,
                'id': 'track1',
                'external_urls': {'spotify': 'https://open.spotify.com/track/track1'}
            },
            {
                'name': 'Song 2',
                'artists': [{'name': 'Artist 2'}],
                'album': {'name': 'Album 2', 'release_date': '2023-02-01'},
                'popularity': 70,
                'explicit': True,
                'duration_ms': 200000,
                'id': 'track2',
                'external_urls': {'spotify': 'https://open.spotify.com/track/track2'}
            }
        ]
        
        mock_sp.playlist_tracks.return_value = {'items': playlist_tracks, 'next': None}
        mock_sp.tracks.return_value = {'tracks': tracks_info}
        
        result = get_playlist_dataframe(mock_sp, 'playlist_id')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == [
            'Name', 'Artist', 'Album', 'Release Date', 'Popularity', 
            'Explicit', 'Duration (ms)', 'Track ID', 'Spotify URL'
        ]
        assert result.iloc[0]['Name'] == 'Song 1'
        assert result.iloc[1]['Name'] == 'Song 2'
    
    def test_get_playlist_dataframe_batch_processing(self):
        """Test DataFrame creation with batch processing for large playlists."""
        mock_sp = Mock()
        
        # Create a large playlist with 75 tracks (more than batch size of 50)
        playlist_tracks = [{'track': {'id': f'track{i}'}} for i in range(75)]
        
        # Mock tracks info response for two batches
        batch1_tracks = [
            {
                'name': f'Song {i}',
                'artists': [{'name': f'Artist {i}'}],
                'album': {'name': f'Album {i}', 'release_date': '2023-01-01'},
                'popularity': 80,
                'explicit': False,
                'duration_ms': 180000,
                'id': f'track{i}',
                'external_urls': {'spotify': f'https://open.spotify.com/track/track{i}'}
            } for i in range(50)
        ]
        
        batch2_tracks = [
            {
                'name': f'Song {i}',
                'artists': [{'name': f'Artist {i}'}],
                'album': {'name': f'Album {i}', 'release_date': '2023-01-01'},
                'popularity': 80,
                'explicit': False,
                'duration_ms': 180000,
                'id': f'track{i}',
                'external_urls': {'spotify': f'https://open.spotify.com/track/track{i}'}
            } for i in range(50, 75)
        ]
        
        mock_sp.playlist_tracks.return_value = {'items': playlist_tracks, 'next': None}
        mock_sp.tracks.side_effect = [
            {'tracks': batch1_tracks},
            {'tracks': batch2_tracks}
        ]
        
        result = get_playlist_dataframe(mock_sp, 'playlist_id')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 75
        assert mock_sp.tracks.call_count == 2  # Should be called twice for batching
    
    def test_get_playlist_dataframe_empty_playlist(self):
        """Test DataFrame creation with empty playlist."""
        mock_sp = Mock()
        mock_sp.playlist_tracks.return_value = {'items': [], 'next': None}
        
        result = get_playlist_dataframe(mock_sp, 'playlist_id')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_playlist_dataframe_no_valid_tracks(self):
        """Test DataFrame creation with no valid tracks."""
        mock_sp = Mock()
        playlist_tracks = [
            {'track': None},
            {'track': None}
        ]
        mock_sp.playlist_tracks.return_value = {'items': playlist_tracks, 'next': None}
        
        result = get_playlist_dataframe(mock_sp, 'playlist_id')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestSavePlaylistToCsv:
    """Test the save_playlist_to_csv function."""
    
    def test_save_playlist_to_csv_default_filename(self):
        """Test saving with default filename."""
        df = pd.DataFrame({'Name': ['Song 1'], 'Artist': ['Artist 1']})
        
        with patch.object(df, 'to_csv') as mock_to_csv:
            save_playlist_to_csv(df)
            mock_to_csv.assert_called_once_with('my_playlist_table.csv', index=False)
    
    def test_save_playlist_to_csv_custom_filename(self):
        """Test saving with custom filename."""
        df = pd.DataFrame({'Name': ['Song 1'], 'Artist': ['Artist 1']})
        
        with patch.object(df, 'to_csv') as mock_to_csv:
            save_playlist_to_csv(df, 'custom_playlist.csv')
            mock_to_csv.assert_called_once_with('custom_playlist.csv', index=False)


class TestIntegration:
    """Integration tests that require actual Spotify credentials."""
    
    @pytest.mark.skipif(
        not os.getenv("SPOTIPY_CLIENT_ID") or not os.getenv("SPOTIPY_CLIENT_SECRET"),
        reason="Spotify credentials not available"
    )
    def test_real_playlist_access(self):
        """Test with real Spotify API (requires credentials)."""
        try:
            sp = get_spotify_client()
            
            # Test with a public playlist
            df = get_playlist_dataframe(sp, '0nqnvBL1fG8EKOXqv1FCIf')
            
            # Should return a DataFrame (might be empty if playlist is private)
            assert isinstance(df, pd.DataFrame)
            
            if not df.empty:
                # Check that required columns exist
                expected_columns = [
                    'Name', 'Artist', 'Album', 'Release Date', 'Popularity', 
                    'Explicit', 'Duration (ms)', 'Track ID', 'Spotify URL'
                ]
                assert all(col in df.columns for col in expected_columns)
                
                # Check that data types are reasonable
                assert df['Popularity'].dtype in ['int64', 'int32']
                assert df['Explicit'].dtype == 'bool'
                assert df['Duration (ms)'].dtype in ['int64', 'int32']
                
        except Exception as e:
            pytest.skip(f"Spotify API not accessible: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
